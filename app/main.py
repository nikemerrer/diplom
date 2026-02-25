# app/main.py
# -*- coding: utf-8 -*-
"""
FastAPI сервис:
- POST /api/jobs                -> создать job (base64 файл) и поставить в очередь (SQLite)
- POST /api/jobs/result         -> получить результат job по job_id (POST, чтобы 1С стабильно читала тело)
- GET  /api/jobs/{job_id}       -> оставить для Swagger/PowerShell (в 1С может быть пустое тело на GET)
- GET  /api/jobs/stats          -> мониторинг очереди/статусов
- GET  /health                  -> healthcheck
"""

import os
import json
import base64
import hashlib
import sqlite3
import threading
import time
import inspect
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ===================== базовые настройки =====================
BASE_DIR = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

CURRENT_MODEL = os.getenv("OLLAMA_MODEL_ID") or os.getenv("MODEL_ID") or "qwen2.5:7b"

app = FastAPI(title="Contract Extractor → 1C JSON (Job Service)")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR) if os.path.isdir(TEMPLATES_DIR) else None

if os.getenv("ENABLE_CORS", "0") == "1":
    allow_origins = os.getenv("ALLOW_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in allow_origins],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ===================== подключение ваших утилит =====================
try:
    from utils import file_reader
except ImportError as e:
    raise RuntimeError("Не найден utils/file_reader.py — проверь структуру проекта.") from e

_to_onec_fn = None
_extract_fn = None
try:
    from utils.entity_extractor import extract_onec_fields as _to_onec_fn  # type: ignore
except Exception:
    pass
try:
    from utils.entity_extractor import extract_entities_from_text as _extract_fn  # type: ignore
except Exception:
    _extract_fn = None

if _to_onec_fn is None and _extract_fn is None:
    raise RuntimeError("Не найден utils/entity_extractor.py с нужными функциями.")

def _read_file_to_text(file_bytes: bytes, filename: Optional[str]) -> str:
    candidates = [
        ("read_file_to_text", {"file_bytes": file_bytes, "filename": filename}),
        ("read_file_to_text", {"file_bytes": file_bytes}),
        ("read_bytes_to_text", {"file_bytes": file_bytes, "filename": filename}),
        ("get_text_from_file", {"file_bytes": file_bytes, "filename": filename}),
    ]
    for name, kwargs in candidates:
        func = getattr(file_reader, name, None)
        if callable(func):
            try:
                sig = inspect.signature(func)
                call_kwargs = {}
                for p in sig.parameters.values():
                    if p.name in kwargs and kwargs[p.name] is not None:
                        call_kwargs[p.name] = kwargs[p.name]
                if not call_kwargs and "file_bytes" in kwargs:
                    return func(kwargs["file_bytes"])  # type: ignore
                return func(**call_kwargs)  # type: ignore
            except Exception:
                continue
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def _process_bytes_to_onec(filename: str, raw_bytes: bytes) -> Dict[str, Any]:
    text = _read_file_to_text(raw_bytes, filename=filename)
    if not text or not text.strip():
        raise RuntimeError("Не удалось извлечь текст из файла")

    if _to_onec_fn:
        onec_result = _to_onec_fn(text)
        if isinstance(onec_result, dict) and "data" in onec_result:
            if onec_result.get("_error"):
                raise RuntimeError(str(onec_result.get("_error")))
            data = onec_result.get("data") or {}
            if not isinstance(data, dict):
                raise RuntimeError("onec_result.data не dict")
            return data
        if isinstance(onec_result, dict):
            return onec_result
        raise RuntimeError("extract_onec_fields вернул не dict")

    if _extract_fn:
        base = _extract_fn(text)
        from utils.entity_extractor import _to_onec_schema  # type: ignore
        data = _to_onec_schema(text, base)  # type: ignore
        if not isinstance(data, dict):
            raise RuntimeError("fallback result не dict")
        return data

    raise RuntimeError("Нет функций экстракции (_to_onec_fn/_extract_fn)")

# ===================== базовые ручки =====================
@app.get("/health", response_class=PlainTextResponse)
async def health():
    return "ok"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if templates:
        return templates.TemplateResponse("index.html", {"request": request, "model_id": CURRENT_MODEL})
    return HTMLResponse("<h3>Сервис запущен. Используйте /docs</h3>")

# ===================== JOB SERVICE (SQLite + Worker) =====================
JOBS_DB_PATH = os.getenv("JOBS_DB_PATH", "jobs.db")
JOBS_POLL_SECONDS = float(os.getenv("JOBS_POLL_SECONDS", "1.0"))

_job_stop = threading.Event()
_job_thread: Optional[threading.Thread] = None

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _db():
    conn = sqlite3.connect(JOBS_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def _init_jobs_db():
    conn = _db()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        status TEXT NOT NULL,
        user_id TEXT,
        doc_id TEXT,
        filename TEXT,
        input_sha256 TEXT NOT NULL,
        result_json TEXT,
        error_text TEXT,
        model_version TEXT,
        attempts INTEGER NOT NULL DEFAULT 0
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS job_payloads (
        job_id TEXT PRIMARY KEY,
        filename TEXT,
        file_bytes BLOB NOT NULL
    );
    """)
    conn.commit()
    conn.close()

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _jobs_worker_loop():
    conn = _db()
    while not _job_stop.is_set():
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT job_id
                FROM jobs
                WHERE status='queued'
                ORDER BY created_at
                LIMIT 1
            """)
            row = cur.fetchone()
            if not row:
                time.sleep(JOBS_POLL_SECONDS)
                continue

            job_id = row["job_id"]

            cur.execute("""
                UPDATE jobs
                SET status='running', started_at=?, attempts=attempts+1, model_version=?
                WHERE job_id=? AND status='queued'
            """, (_utc_now(), CURRENT_MODEL, job_id))
            conn.commit()

            cur.execute("SELECT filename, file_bytes FROM job_payloads WHERE job_id=?", (job_id,))
            payload = cur.fetchone()
            if not payload:
                raise RuntimeError("Payload not found")

            filename = payload["filename"] or ""
            file_bytes = payload["file_bytes"]

            result = _process_bytes_to_onec(filename, file_bytes)
            result_json = json.dumps(result, ensure_ascii=False)

            cur.execute("""
                UPDATE jobs
                SET status='done', finished_at=?, result_json=?, error_text=NULL
                WHERE job_id=? AND status='running'
            """, (_utc_now(), result_json, job_id))
            conn.commit()

        except Exception as e:
            try:
                err = str(e)
                cur = conn.cursor()
                cur.execute("""
                    UPDATE jobs
                    SET status='error', finished_at=?, error_text=?
                    WHERE job_id=? AND status='running'
                """, (_utc_now(), err, job_id))
                conn.commit()
            except Exception:
                pass
            time.sleep(0.2)

    conn.close()

class JobMeta(BaseModel):
    user_id: Optional[str] = None
    doc_id: Optional[str] = None

class CreateJobRequest(BaseModel):
    filename: str = Field(default="")
    content_base64: str
    meta: Optional[JobMeta] = None

class JobResultRequest(BaseModel):
    job_id: str

@app.on_event("startup")
def _jobs_startup():
    _init_jobs_db()
    global _job_thread
    _job_thread = threading.Thread(target=_jobs_worker_loop, daemon=True)
    _job_thread.start()

@app.on_event("shutdown")
def _jobs_shutdown():
    _job_stop.set()

@app.post("/api/jobs")
def api_create_job(req: CreateJobRequest):
    if not req.content_base64 or len(req.content_base64) < 16:
        raise HTTPException(status_code=400, detail="content_base64 is empty")

    try:
        file_bytes = base64.b64decode("".join(req.content_base64.split()))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64")

    file_hash = _sha256(file_bytes)
    job_id = str(__import__("uuid").uuid4())
    created_at = _utc_now()

    user_id = req.meta.user_id if req.meta else None
    doc_id = req.meta.doc_id if req.meta else None

    conn = _db()
    cur = conn.cursor()

    cur.execute("""
        SELECT job_id FROM jobs
        WHERE input_sha256=? AND status='done'
        ORDER BY finished_at DESC
        LIMIT 1
    """, (file_hash,))
    cached = cur.fetchone()
    if cached:
        conn.close()
        return {"job_id": cached["job_id"], "status": "done"}

    cur.execute("""
        INSERT INTO jobs(job_id, created_at, status, user_id, doc_id, filename, input_sha256, model_version)
        VALUES (?, ?, 'queued', ?, ?, ?, ?, ?)
    """, (job_id, created_at, user_id, doc_id, req.filename, file_hash, CURRENT_MODEL))

    cur.execute("""
        INSERT INTO job_payloads(job_id, filename, file_bytes)
        VALUES (?, ?, ?)
    """, (job_id, req.filename, sqlite3.Binary(file_bytes)))

    conn.commit()
    conn.close()

    return {"job_id": job_id, "status": "queued"}

def _job_row_to_payload(row: sqlite3.Row) -> Dict[str, Any]:
    result = json.loads(row["result_json"]) if row["result_json"] else None
    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "result": result,
        "error": row["error_text"],
        "model_version": row["model_version"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
    }

@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str):
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="job not found")

    payload = _job_row_to_payload(row)
    body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    return Response(
        content=body_bytes,
        media_type="application/json; charset=utf-8",
        headers={
            "Content-Length": str(len(body_bytes)),
            "Connection": "close",
        },
    )

@app.post("/api/jobs/result")
def api_get_job_result(req: JobResultRequest):
    job_id = req.job_id

    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="job not found")

    payload = _job_row_to_payload(row)
    body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    return Response(
        content=body_bytes,
        media_type="application/json; charset=utf-8",
        headers={
            "Content-Length": str(len(body_bytes)),
            "Connection": "close",
        },
    )

@app.get("/api/jobs/stats")
def api_jobs_stats():
    conn = _db()
    cur = conn.cursor()
    out = {}
    for st in ("queued", "running", "done", "error"):
        cur.execute("SELECT COUNT(1) AS c FROM jobs WHERE status=?", (st,))
        out[st] = int(cur.fetchone()["c"])

    cur.execute("SELECT MIN(created_at) AS m FROM jobs WHERE status='queued'")
    out["oldest_queued_created_at"] = cur.fetchone()["m"]
    conn.close()
    return out