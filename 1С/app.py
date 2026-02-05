from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from uuid import uuid4
import base64
import json

app = FastAPI()

STORAGE = Path("storage")
STORAGE.mkdir(exist_ok=True)

# Кэш в памяти (не обязателен, но ускоряет)
JOBS: dict[str, dict] = {}

app.mount("/files", StaticFiles(directory=str(STORAGE)), name="files")


def _job_dir(job_id: str) -> Path:
    return STORAGE / job_id


def _meta_path(job_id: str) -> Path:
    return _job_dir(job_id) / "meta.json"


def save_meta(job_id: str, meta: dict) -> None:
    p = _meta_path(job_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def load_meta(job_id: str) -> dict | None:
    p = _meta_path(job_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


class CreateJobReq(BaseModel):
    filename: str
    content_base64: str


class CommentReq(BaseModel):
    comment: str


@app.post("/api/jobs")
def create_job(req: CreateJobReq):
    try:
        data = base64.b64decode(req.content_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64")

    job_id = str(uuid4())
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    file_path = job_dir / req.filename
    file_path.write_bytes(data)

    meta = {
        "filename": req.filename,
        "file_rel": f"{job_id}/{req.filename}",
        "comment": "",
        "status": "uploaded",
    }

    JOBS[job_id] = meta
    save_meta(job_id, meta)

    return {
        "job_id": job_id,
        "web_url": f"http://127.0.0.1:8000/jobs/{job_id}",
        "api_url": f"http://127.0.0.1:8000/api/jobs/{job_id}",
        "file_url": f"http://127.0.0.1:8000/files/{job_id}/{req.filename}",
    }


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(job_id: str):
    job = JOBS.get(job_id) or load_meta(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    JOBS[job_id] = job  # кэшируем

    file_url = f"/files/{job['file_rel']}"
    comment = job.get("comment", "")

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Job {job_id}</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 24px auto; }}
    textarea {{ width: 100%; height: 180px; }}
    .box {{ border: 1px solid #ccc; padding: 12px; border-radius: 8px; }}
    .row {{ margin: 12px 0; }}
    button {{ padding: 8px 12px; }}
    code {{ background:#f5f5f5; padding:2px 4px; border-radius:4px; }}
  </style>
</head>
<body>
  <h2>Заявка: <code>{job_id}</code></h2>

  <div class="box">
    <div class="row">
      Файл: <b>{job['filename']}</b> —
      <a href="{file_url}" target="_blank">скачать/открыть</a>
    </div>

    <div class="row">
      <label>Контекст/комментарий (то, что должно попасть в 1С):</label><br>
      <textarea id="comment">{comment}</textarea>
    </div>

    <div class="row">
      <button onclick="send()">Отправить</button>
      <span id="status"></span>
    </div>
  </div>

<script>
async function send() {{
  const comment = document.getElementById("comment").value;
  const res = await fetch("/api/jobs/{job_id}/comment", {{
    method: "POST",
    headers: {{ "Content-Type": "application/json" }},
    body: JSON.stringify({{ comment }})
  }});
  const status = document.getElementById("status");
  if (!res.ok) {{
    status.textContent = "Ошибка сохранения";
    return;
  }}
  status.textContent = "Сохранено. Теперь нажмите в 1С «Получить».";
}}
</script>
</body>
</html>
"""
    return HTMLResponse(html)


@app.post("/api/jobs/{job_id}/comment")
def set_comment(job_id: str, req: CommentReq):
    job = JOBS.get(job_id) or load_meta(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    job["comment"] = req.comment
    job["status"] = "commented"

    JOBS[job_id] = job
    save_meta(job_id, job)

    return {"ok": True}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id) or load_meta(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    JOBS[job_id] = job

    return {
        "job_id": job_id,
        "status": job.get("status", ""),
        "filename": job.get("filename", ""),
        "comment": job.get("comment", ""),
    }
