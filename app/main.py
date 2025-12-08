# app/main.py
# -*- coding: utf-8 -*-
import os
import io
import json
import inspect
from typing import Any, Dict, Optional, Tuple, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="Contract Extractor → 1C JSON")

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

# --- Файл чтения ---
try:
    from utils import file_reader
except ImportError as e:
    raise RuntimeError("Не найден utils/file_reader.py — проверь структуру проекта.") from e

# --- Экстракция и маппинг в 1С ---
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

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    return HTMLResponse("<h3>Загрузи документ: POST /</h3>")

@app.get("/health", response_class=PlainTextResponse)
async def health():
    return "ok"

LAST_JSON_RESULT: Optional[Dict[str, Any]] = None

@app.post("/", response_class=HTMLResponse)
async def upload_via_root(request: Request, file: UploadFile = File(...)):
    try:
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Пустой файл.")

        text = _read_file_to_text(raw_bytes, filename=file.filename)
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Не удалось получить текст из файла.")

        # --- 1) Генерим 1С-JSON ---
        if _to_onec_fn:
            onec_result = _to_onec_fn(text)
            if isinstance(onec_result, dict) and "data" in onec_result:
                onec_json = onec_result.get("data", {})
                debug_raw = onec_result.get("_debug_raw")
            else:
                onec_json = onec_result
                debug_raw = None
        elif _extract_fn:
            base = _extract_fn(text)
            # локальный fallback: простое отображение
            from utils.entity_extractor import _to_onec_schema  # type: ignore
            onec_json = _to_onec_schema(text, base)  # type: ignore
            debug_raw = None
        else:
            onec_json = {}
            debug_raw = None

        pretty = json.dumps(onec_json, ensure_ascii=False, indent=2)
        preview_chars = int(os.getenv("PREVIEW_CHARS", "6000"))
        preview_text = text[:preview_chars]

        payload = {
            "ok": True,
            "errors": [],
            "data": onec_json,     # <<< именно 1С-ориентированный JSON
            "filename": file.filename,
            "length": len(text),
            "debug_raw": debug_raw,
        }

        global LAST_JSON_RESULT
        LAST_JSON_RESULT = payload

        if templates:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "filename": file.filename, "text": preview_text, "json_str": pretty, "debug_raw": debug_raw},
            )
        return JSONResponse(payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {e}")

@app.post("/api/extract-file")
async def extract_file(file: UploadFile = File(...)):
    try:
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Пустой файл.")
        text = _read_file_to_text(raw_bytes, filename=file.filename)
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Не удалось получить текст из файла.")
        onec_result = _to_onec_fn(text) if _to_onec_fn else {}
        if isinstance(onec_result, dict) and "data" in onec_result:
            onec_json = onec_result.get("data", {})
            debug_raw = onec_result.get("_debug_raw")
        else:
            onec_json = onec_result
            debug_raw = None
        payload = {"ok": True, "errors": [], "data": onec_json, "filename": file.filename, "length": len(text), "debug_raw": debug_raw}
        global LAST_JSON_RESULT
        LAST_JSON_RESULT = payload
        return JSONResponse(payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {e}")

@app.post("/download-json")
async def download_json_post(request: Request, filename: Optional[str] = None):
    try:
        form = await request.form()
        json_payload = form.get("json_payload")
        if json_payload:
            content_bytes = json_payload.encode("utf-8")
        elif LAST_JSON_RESULT and "data" in LAST_JSON_RESULT:
            content_bytes = json.dumps(LAST_JSON_RESULT["data"], ensure_ascii=False, indent=2).encode("utf-8")
        else:
            raise HTTPException(status_code=404, detail="Нет данных для выгрузки.")
        fname = filename or "onec_fields.json"
        headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
        return StreamingResponse(io.BytesIO(content_bytes), media_type="application/json", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при скачивании JSON: {e}")

@app.get("/download-json")
async def download_json_get(filename: Optional[str] = None):
    if not LAST_JSON_RESULT or "data" not in LAST_JSON_RESULT:
        raise HTTPException(status_code=404, detail="Нет данных для выгрузки.")
    content_bytes = json.dumps(LAST_JSON_RESULT["data"], ensure_ascii=False, indent=2).encode("utf-8")
    fname = filename or "onec_fields.json"
    headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
    return StreamingResponse(io.BytesIO(content_bytes), media_type="application/json", headers=headers)
