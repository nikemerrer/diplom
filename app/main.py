# app/main.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DB_PATH = os.getenv("DB_PATH", "jobs.db")
POLL_INTERVAL_SEC = float(os.getenv("JOB_POLL_INTERVAL_SEC", "1.0"))
PREVIEW_CHARS = int(os.getenv("PREVIEW_CHARS", "6000"))
MODEL_ID = os.getenv("OLLAMA_MODEL_ID") or os.getenv("MODEL_ID") or "qwen2.5:1.5b"

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="1C AI Job Service")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            created_at TEXT,
            started_at TEXT,
            finished_at TEXT,
            status TEXT,
            filename TEXT,
            result_json TEXT,
            error_text TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS payloads (
            job_id TEXT PRIMARY KEY,
            file_bytes BLOB
        )
        """
    )
    conn.commit()
    conn.close()


class CreateJob(BaseModel):
    filename: str
    content_base64: str
    meta: Optional[Dict[str, Any]] = None


class GetResult(BaseModel):
    job_id: str


stop_event = threading.Event()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def seconds_since(value: Optional[str]) -> Optional[float]:
    dt = parse_ts(value)
    if dt is None:
        return None
    return max((datetime.now(timezone.utc) - dt).total_seconds(), 0.0)


def seconds_between(start: Optional[str], end: Optional[str]) -> Optional[float]:
    start_dt = parse_ts(start)
    end_dt = parse_ts(end)
    if start_dt is None or end_dt is None:
        return None
    return max((end_dt - start_dt).total_seconds(), 0.0)


def trim_text(value: Optional[str], limit: int = 160) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def parse_result_json(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_result_meta(raw: Optional[str]) -> Dict[str, Any]:
    result = parse_result_json(raw) or {}
    meta = result.get("_meta")
    return meta if isinstance(meta, dict) else {}


def render_json_payload(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def process_file_bytes(file_bytes: bytes, filename: str = "") -> Dict[str, Any]:
    from utils import file_reader
    from utils.entity_extractor import extract_onec_fields

    text = file_reader.read_file_to_text(file_bytes=file_bytes, filename=filename)
    if not text or not text.strip():
        raise RuntimeError("Не удалось извлечь текст из файла")

    result = extract_onec_fields(text)
    if not isinstance(result, dict):
        raise RuntimeError("extract_onec_fields вернул некорректный результат")

    return result


def collect_stats(cur: sqlite3.Cursor) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    total = 0
    for status_name in ["queued", "running", "done", "error"]:
        cur.execute("SELECT COUNT(*) AS c FROM jobs WHERE status=?", (status_name,))
        count = int(cur.fetchone()["c"])
        data[status_name] = count
        total += count

    cur.execute(
        "SELECT created_at FROM jobs WHERE status='queued' ORDER BY created_at ASC LIMIT 1"
    )
    oldest_queued = cur.fetchone()
    cur.execute(
        "SELECT started_at FROM jobs WHERE status='running' ORDER BY started_at ASC LIMIT 1"
    )
    oldest_running = cur.fetchone()

    data["total"] = total
    data["oldest_queued_age_sec"] = seconds_since(
        oldest_queued["created_at"] if oldest_queued else None
    )
    data["oldest_running_age_sec"] = seconds_since(
        oldest_running["started_at"] if oldest_running else None
    )
    return data


def worker() -> None:
    conn = db()

    while not stop_event.is_set():
        cur = conn.cursor()
        cur.execute(
            "SELECT job_id FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
        )
        row = cur.fetchone()

        if not row:
            time.sleep(POLL_INTERVAL_SEC)
            continue

        job_id = row["job_id"]
        cur.execute(
            "UPDATE jobs SET status='running', started_at=? WHERE job_id=?",
            (now(), job_id),
        )
        conn.commit()

        try:
            cur.execute(
                """
                SELECT p.file_bytes, j.filename
                FROM payloads p
                JOIN jobs j ON j.job_id = p.job_id
                WHERE p.job_id=?
                """,
                (job_id,),
            )
            payload = cur.fetchone()
            if not payload:
                raise RuntimeError(f"Payload для job_id={job_id} не найден")

            result = process_file_bytes(payload["file_bytes"], payload["filename"] or "")
            cur.execute(
                "UPDATE jobs SET status='done', finished_at=?, result_json=? WHERE job_id=?",
                (now(), json.dumps(result, ensure_ascii=False), job_id),
            )
        except Exception as exc:
            cur.execute(
                "UPDATE jobs SET status='error', finished_at=?, error_text=? WHERE job_id=?",
                (now(), str(exc), job_id),
            )

        conn.commit()

    conn.close()


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "filename": "",
            "text": "",
            "json_str": "",
            "debug_raw": "",
            "model_id": MODEL_ID,
        },
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    conn = db()
    cur = conn.cursor()
    stats = collect_stats(cur)
    conn.close()
    return {"status": "ok", "model_id": MODEL_ID, "jobs": stats}


@app.post("/api/extract-file")
async def extract_file(file: UploadFile = File(...)) -> JSONResponse:
    from utils import file_reader

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty file payload")

    try:
        preview_text = file_reader.read_file_to_text(payload, file.filename)
        result = process_file_bytes(payload, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    meta = result.get("_meta") if isinstance(result.get("_meta"), dict) else {}
    return JSONResponse(
        {
            "filename": file.filename,
            "preview_text": preview_text[:PREVIEW_CHARS],
            "data": result,
            "debug_raw": render_json_payload(meta) if meta else "",
        }
    )


@app.post("/download-json")
def download_json(json_payload: str = Form(...)) -> Response:
    try:
        parsed = json.loads(json_payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}")

    body = render_json_payload(parsed).encode("utf-8")
    return Response(
        content=body,
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="onec_fields.json"'},
    )


@app.get("/api/jobs/stats")
def stats() -> Dict[str, Any]:
    conn = db()
    cur = conn.cursor()
    data = collect_stats(cur)
    conn.close()
    return data


@app.post("/api/jobs")
def create_job(req: CreateJob):
    def normalize_base64(s: str) -> str:
        s = (s or "").strip()
        s = s.replace("\r", "").replace("\n", "").replace("\t", "").replace(" ", "")
        pad = len(s) % 4
        if pad:
            s += "=" * (4 - pad)
        return s

    raw_b64 = normalize_base64(req.content_base64)
    try:
        file_bytes = base64.b64decode(raw_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {exc}")

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file payload")

    job_id = str(uuid.uuid4())
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO jobs VALUES (?, ?, NULL, NULL, 'queued', ?, NULL, NULL)",
        (job_id, now(), req.filename),
    )
    cur.execute("INSERT INTO payloads VALUES (?, ?)", (job_id, sqlite3.Binary(file_bytes)))
    conn.commit()
    conn.close()
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/jobs/result")
def get_result(req: GetResult) -> Response:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs WHERE job_id=?", (req.job_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Not found")

    payload = {
        "job_id": row["job_id"],
        "status": row["status"],
        "result": json.loads(row["result_json"]) if row["result_json"] else None,
        "error": row["error_text"],
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Length": str(len(body)), "Connection": "close"},
    )


@app.get("/api/jobs")
def list_jobs(status: Optional[str] = None, limit: int = 50, offset: int = 0):
    limit = max(1, min(limit, 500))
    offset = max(0, offset)

    conn = db()
    cur = conn.cursor()
    cols = "job_id, created_at, started_at, finished_at, status, filename, error_text"
    if status:
        cur.execute(
            f"SELECT {cols} FROM jobs WHERE status=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (status, limit, offset),
        )
    else:
        cur.execute(
            f"SELECT {cols} FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )

    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/jobs/{job_id}")
def get_job_detail(job_id: str) -> Dict[str, Any]:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Not found")

    return {
        "job_id": row["job_id"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "status": row["status"],
        "filename": row["filename"],
        "result": json.loads(row["result_json"]) if row["result_json"] else None,
        "error": row["error_text"],
    }


@app.get("/api/monitor/summary")
def monitor_summary(limit: int = 120) -> Dict[str, Any]:
    limit = max(10, min(limit, 300))
    conn = db()
    cur = conn.cursor()

    stats_data = collect_stats(cur)

    cur.execute(
        """
        SELECT job_id, created_at, started_at, finished_at, status, filename, result_json, error_text
        FROM jobs
        WHERE status='running'
        ORDER BY started_at ASC
        LIMIT ?
        """,
        (limit,),
    )
    running_rows = cur.fetchall()

    cur.execute(
        """
        SELECT job_id, created_at, started_at, finished_at, status, filename
        FROM jobs
        WHERE status='queued'
        ORDER BY created_at ASC
        LIMIT ?
        """,
        (limit,),
    )
    queue_rows = cur.fetchall()

    cur.execute(
        """
        SELECT job_id, created_at, started_at, finished_at, status, filename, result_json
        FROM jobs
        WHERE status='done'
        ORDER BY finished_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    done_rows = cur.fetchall()

    cur.execute(
        """
        SELECT job_id, created_at, started_at, finished_at, status, filename, error_text
        FROM jobs
        WHERE status='error'
        ORDER BY finished_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    error_rows = cur.fetchall()
    conn.close()

    running_jobs = [
        {
            "job_id": row["job_id"],
            "filename": row["filename"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "processing_sec": seconds_since(row["started_at"]),
            "meta": extract_result_meta(row["result_json"]),
        }
        for row in running_rows
    ]

    queue_preview = [
        {
            "job_id": row["job_id"],
            "filename": row["filename"],
            "created_at": row["created_at"],
            "wait_sec": seconds_since(row["created_at"]),
            "queue_position": index + 1,
        }
        for index, row in enumerate(queue_rows)
    ]

    recent_done = [
        {
            "job_id": row["job_id"],
            "filename": row["filename"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "wait_sec": seconds_between(row["created_at"], row["started_at"]),
            "processing_sec": seconds_between(row["started_at"], row["finished_at"]),
            "meta": extract_result_meta(row["result_json"]),
        }
        for row in done_rows
    ]

    recent_error = [
        {
            "job_id": row["job_id"],
            "filename": row["filename"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "error": row["error_text"],
            "error_preview": trim_text(row["error_text"]),
        }
        for row in error_rows
    ]

    processing_values = [
        item["processing_sec"]
        for item in recent_done[:20]
        if isinstance(item["processing_sec"], (int, float))
    ]
    avg_processing = (
        sum(processing_values) / len(processing_values) if processing_values else None
    )

    return {
        "stats": stats_data,
        "active_job": running_jobs[0] if running_jobs else None,
        "running_jobs": running_jobs,
        "queue_preview": queue_preview,
        "recent_done": recent_done,
        "recent_error": recent_error,
        "avg_processing_sec_recent": avg_processing,
        "model_id": MODEL_ID,
    }


@app.get("/monitor", response_class=HTMLResponse)
def monitor_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "monitor.html", {})


@app.on_event("startup")
def startup() -> None:
    init_db()
    stop_event.clear()
    threading.Thread(target=worker, daemon=True).start()

    print("\n========================================")
    print("Server started")
    print(f"Web UI:   http://localhost:{PORT}/")
    print(f"Monitor:  http://localhost:{PORT}/monitor")
    print("========================================\n")


@app.on_event("shutdown")
def shutdown() -> None:
    stop_event.set()
