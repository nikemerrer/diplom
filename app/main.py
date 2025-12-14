# app/main.py
# -*- coding: utf-8 -*-
import os
import io
import sys
import html
import json
import inspect
import tempfile
from typing import Any, Dict, Optional, Tuple, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
PREVIEW_CHARS = int(os.getenv("PREVIEW_CHARS", "6000"))

# Включаем локальную копию langextract, если pip-модуля нет (например, папка ../langextract-main)
try:
    import importlib.util as _importlib_util

    if _importlib_util.find_spec("langextract") is None:
        POSSIBLE_LX_DIRS = [
            os.path.abspath(os.path.join(BASE_DIR, "..", "langextract-main")),
            os.path.abspath(os.path.join(BASE_DIR, "..", "langextract")),
        ]
        for _p in POSSIBLE_LX_DIRS:
            if os.path.isdir(_p) and _p not in sys.path:
                sys.path.insert(0, _p)
except Exception:
    pass

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


def _import_langextract():
    """
    Ленивый импорт langextract из локальной папки или окружения.
    Если нет pandas, подставляем простой заглушечный модуль, чтобы сработали io/visualize.
    """
    try:
        import langextract as lx  # type: ignore
        return lx
    except ModuleNotFoundError as e:
        if e.name == "pandas":
            import types

            def _missing_pandas(*args, **kwargs):
                raise ModuleNotFoundError("pandas не установлен; установите его или добавьте в зависимости.")

            dummy_pd = types.SimpleNamespace(read_csv=_missing_pandas, DataFrame=None)
            sys.modules["pandas"] = dummy_pd
            # tqdm заглушка
            dummy_tqdm_mod = types.SimpleNamespace()
            class _DummyTqdm:
                def __init__(self, *a, **k): pass
                def update(self, *a, **k): pass
                def close(self): pass
            dummy_tqdm_mod.tqdm = _DummyTqdm
            sys.modules.setdefault("tqdm", dummy_tqdm_mod)
            # absl.logging заглушка
            import logging
            class _AbslLogger:
                def __init__(self):
                    self._log = logging.getLogger("absl")
                def debug(self, *a, **k): self._log.debug(*a, **k)
                def info(self, *a, **k): self._log.info(*a, **k)
                def warning(self, *a, **k): self._log.warning(*a, **k)
                def error(self, *a, **k): self._log.error(*a, **k)
                def fatal(self, *a, **k): self._log.critical(*a, **k)
            absl_mod = types.SimpleNamespace(logging=_AbslLogger())
            sys.modules.setdefault("absl", absl_mod)
            sys.modules.setdefault("absl.logging", absl_mod.logging)

            import langextract as lx  # type: ignore
            return lx
        raise

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
LAST_TEXT: Optional[str] = None

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
        preview_text = text[:PREVIEW_CHARS]

        payload = {
            "ok": True,
            "errors": [],
            "data": onec_json,     # <<< именно 1С-ориентированный JSON
            "filename": file.filename,
            "length": len(text),
            "debug_raw": debug_raw,
            "preview_text": preview_text,
        }

        global LAST_JSON_RESULT
        LAST_JSON_RESULT = payload
        global LAST_TEXT
        LAST_TEXT = text

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
        preview_text = text[:PREVIEW_CHARS]
        payload = {
            "ok": True,
            "errors": [],
            "data": onec_json,
            "filename": file.filename,
            "length": len(text),
            "debug_raw": debug_raw,
            "preview_text": preview_text,
        }
        global LAST_JSON_RESULT
        LAST_JSON_RESULT = payload
        global LAST_TEXT
        LAST_TEXT = text
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


@app.get("/visualization", response_class=HTMLResponse)
async def visualization():
    if not LAST_JSON_RESULT or "data" not in LAST_JSON_RESULT:
        raise HTTPException(status_code=404, detail="Нет данных для визуализации. Сначала загрузите документ.")
    if LAST_TEXT is None:
        raise HTTPException(status_code=404, detail="Нет текста документа для визуализации.")

    attrs = LAST_JSON_RESULT.get("data") or {}
    title = ""
    if isinstance(attrs, dict):
        title = attrs.get("заголовок") or attrs.get("title") or ""
    title = title or LAST_JSON_RESULT.get("filename") or "Документ"
    text = LAST_TEXT or ""
    if not text:
        raise HTTPException(status_code=400, detail="Пустой текст, нечего визуализировать.")

    try:
        lx = _import_langextract()
        lx_data = lx.data
        lx_io = lx.io
        extraction = lx_data.Extraction(
            extraction_class="onec_fields",
            extraction_text=str(title),
            attributes=attrs if isinstance(attrs, dict) else {},
            char_interval=lx_data.CharInterval(start_pos=0, end_pos=max(1, len(text))),
        )
        annotated = lx_data.AnnotatedDocument(
            document_id=str(LAST_JSON_RESULT.get("filename") or "doc"),
            text=text,
            extractions=[extraction],
        )
        with tempfile.TemporaryDirectory(prefix="lxviz_") as tmpdir:
            lx_io.save_annotated_documents([annotated], output_dir=tmpdir, output_name="extraction_results.jsonl", show_progress=False)
            jsonl_path = os.path.join(tmpdir, "extraction_results.jsonl")
            html_content = lx.visualize(jsonl_path, show_legend=True, gif_optimized=True)
        if hasattr(html_content, "data"):
            html_content = html_content.data  # type: ignore[attr-defined]
        return HTMLResponse(content=html_content)
    except HTTPException:
        raise
    except Exception as e:
        # Fallback: простая HTML-страница без langextract (например, отсутствуют зависимости)
        fallback_html = f"""
        <html>
        <head>
          <meta charset='utf-8'>
          <title>Визуализация (fallback)</title>
          <style>
            body {{ background:#0b1224; color:#e5e7eb; font-family: 'Manrope', 'Segoe UI', sans-serif; margin:0; padding:24px; }}
            .wrap {{ max-width: 1000px; margin: 0 auto; }}
            .header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; }}
            .pill {{ padding:6px 10px; border-radius:999px; background:rgba(255,255,255,0.08); color:#cbd5e1; font-size:12px; }}
            .panel {{ background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:16px; margin-bottom:14px; }}
            pre {{ background:#0a0f1f; border:1px solid rgba(255,255,255,0.08); padding:12px; border-radius:12px; overflow:auto; }}
            textarea {{ width:100%; min-height:320px; border-radius:12px; border:1px solid rgba(255,255,255,0.08); background:#0d152b; color:#e5e7eb; padding:12px; }}
            .muted {{ color:#94a3b8; font-size:13px; }}
          </style>
        </head>
        <body>
          <div class="wrap">
            <div class="header">
              <h2 style="margin:0">Визуализация (упрощённая)</h2>
              <span class="pill">LangExtract недоступен: {html.escape(str(e))}</span>
            </div>
            <div class="panel">
              <p class="muted">Показываем текст и извлечённые атрибуты. Для полноценной подсветки установите зависимости langextract (pandas, absl, tqdm и др.).</p>
            </div>
            <div class="panel">
              <h3 style="margin-top:0">Текст</h3>
              <textarea readonly>{html.escape(text)}</textarea>
            </div>
            <div class="panel">
              <h3 style="margin-top:0">JSON</h3>
              <pre>{html.escape(json.dumps(attrs, ensure_ascii=False, indent=2))}</pre>
            </div>
          </div>
        </body>
        </html>
        """
        return HTMLResponse(content=fallback_html)
