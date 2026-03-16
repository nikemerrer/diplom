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
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel


# =========================================================
# НАСТРОЙКИ
# =========================================================
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DB_PATH = os.getenv("DB_PATH", "jobs.db")
POLL_INTERVAL_SEC = float(os.getenv("JOB_POLL_INTERVAL_SEC", "1.0"))

app = FastAPI(title="1C AI Job Service")


# =========================================================
# БАЗА ДАННЫХ
# =========================================================
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


# =========================================================
# МОДЕЛИ API
# =========================================================
class CreateJob(BaseModel):
    filename: str
    content_base64: str
    meta: Optional[Dict[str, Any]] = None


class GetResult(BaseModel):
    job_id: str


# =========================================================
# ВСПОМОГАТЕЛЬНОЕ
# =========================================================
stop_event = threading.Event()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


# =========================================================
# WORKER
# =========================================================
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

            result = process_file_bytes(
                payload["file_bytes"],
                payload["filename"] or "",
            )

            cur.execute(
                "UPDATE jobs SET status='done', finished_at=?, result_json=? WHERE job_id=?",
                (now(), json.dumps(result, ensure_ascii=False), job_id),
            )
        except Exception as e:
            cur.execute(
                "UPDATE jobs SET status='error', finished_at=?, error_text=? WHERE job_id=?",
                (now(), str(e), job_id),
            )

        conn.commit()

    conn.close()


# =========================================================
# API
# =========================================================
@app.get("/api/jobs/stats")
def stats() -> Dict[str, int]:
    conn = db()
    cur = conn.cursor()

    data: Dict[str, int] = {}
    total = 0
    for status_name in ["queued", "running", "done", "error"]:
        cur.execute("SELECT COUNT(*) AS c FROM jobs WHERE status=?", (status_name,))
        count = int(cur.fetchone()["c"])
        data[status_name] = count
        total += count

    data["total"] = total
    conn.close()
    return data


@app.post("/api/jobs")
def create_job(req: CreateJob):
    def normalize_base64(s: str) -> str:
        s = (s or "").strip()
        s = s.replace("\r", "").replace("\n", "").replace("\t", "").replace(" ", "")
        # если 1С прислала строку без добивки "="
        pad = len(s) % 4
        if pad:
            s += "=" * (4 - pad)
        return s

    raw_b64 = normalize_base64(req.content_base64)

    try:
        file_bytes = base64.b64decode(raw_b64)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64: {e}"
        )

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file payload")

    job_id = str(uuid.uuid4())

    conn = db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO jobs VALUES (?, ?, NULL, NULL, 'queued', ?, NULL, NULL)",
        (job_id, now(), req.filename),
    )
    cur.execute(
        "INSERT INTO payloads VALUES (?, ?)",
        (job_id, sqlite3.Binary(file_bytes)),
    )
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
        headers={
            "Content-Length": str(len(body)),
            "Connection": "close",
        },
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





# =========================================================
# СТРАНИЦА МОНИТОРИНГА
# =========================================================
@app.get("/monitor", response_class=HTMLResponse)
def monitor_page() -> str:
    return """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Job Monitor</title>
  <style>
    :root{
      --bg:#0b1020; --card:#111833; --text:#e7ecff; --muted:#9aa6d3;
      --line:#24305e; --good:#2ecc71; --warn:#f1c40f; --bad:#e74c3c; --info:#4aa3ff;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Noto Sans", "Liberation Sans", sans-serif;
    }
    body{margin:0;background:linear-gradient(180deg,#070b16,#0b1020);color:var(--text);font-family:var(--sans);}
    .wrap{max-width:1200px;margin:28px auto;padding:0 16px;}
    .top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:16px;}
    h1{font-size:22px;margin:0;font-weight:800;letter-spacing:.2px}
    .card{background:rgba(17,24,51,.82);border:1px solid var(--line);border-radius:16px;box-shadow:0 10px 30px rgba(0,0,0,.25)}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;}
    @media (max-width:900px){.grid{grid-template-columns:1fr}}
    .pad{padding:14px 16px;}
    .row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
    .muted{color:var(--muted);font-size:13px}
    .badge{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);padding:6px 10px;border-radius:999px;font-weight:700;font-size:13px}
    .dot{width:8px;height:8px;border-radius:50%}
    .dot.good{background:var(--good)} .dot.warn{background:var(--warn)}
    .dot.bad{background:var(--bad)} .dot.info{background:var(--info)}
    .controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
    select, input, button{background:#0d1430;color:var(--text);border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-size:14px;}
    button{cursor:pointer;font-weight:800}
    button.primary{background:#143a7a;border-color:#1f4da0}
    button.danger{background:#4a1820;border-color:#7a1f2f}
    button:disabled{opacity:.55;cursor:not-allowed}
    table{width:100%;border-collapse:separate;border-spacing:0;overflow:hidden;border-radius:16px}
    thead th{position:sticky;top:0;background:#0f1736;border-bottom:1px solid var(--line);text-align:left;padding:12px 10px;font-size:13px;color:var(--muted)}
    tbody td{border-bottom:1px solid rgba(36,48,94,.55);padding:12px 10px;font-size:14px;vertical-align:top}
    tbody tr:hover{background:rgba(255,255,255,.03)}
    .mono{font-family:var(--mono);font-size:12.5px}
    .status{font-weight:900}
    .status.done{color:var(--good)}
    .status.running{color:var(--info)}
    .status.queued{color:var(--warn)}
    .status.error{color:var(--bad)}
    .pill{display:inline-block;padding:4px 10px;border-radius:999px;border:1px solid var(--line);font-size:12px}
    .right{margin-left:auto}
    .small{font-size:12px}
    .modal{position:fixed;inset:0;background:rgba(0,0,0,.6);display:none;align-items:center;justify-content:center;padding:16px}
    .modal.open{display:flex}
    .modal .box{max-width:900px;width:100%;max-height:85vh;overflow:auto}
    pre{margin:0;white-space:pre-wrap;word-break:break-word;font-family:var(--mono);font-size:12.5px;color:#dfe7ff}
    .footer{margin-top:10px;display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}
    a{color:#9cc3ff}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <h1>Job Monitor</h1>
      <div class="muted" id="updated">—</div>
    </div>

    <div class="grid">
      <div class="card pad">
        <div class="row">
          <div class="badge"><span class="dot warn"></span> queued: <span id="stQueued">0</span></div>
          <div class="badge"><span class="dot info"></span> running: <span id="stRunning">0</span></div>
          <div class="badge"><span class="dot good"></span> done: <span id="stDone">0</span></div>
          <div class="badge"><span class="dot bad"></span> error: <span id="stError">0</span></div>
          <div class="right muted small">total: <span id="stTotal">0</span></div>
        </div>
        <div class="footer muted small">
          <div>Автообновление: <span id="autoState">ON</span></div>
          <div>API: <a href="/docs" target="_blank">/docs</a> • <a href="/api/jobs/stats" target="_blank">/api/jobs/stats</a></div>
        </div>
      </div>

      <div class="card pad">
        <div class="controls">
          <label class="muted">Статус:</label>
          <select id="fStatus">
            <option value="">Все</option>
            <option value="queued">queued</option>
            <option value="running">running</option>
            <option value="done">done</option>
            <option value="error">error</option>
          </select>

          <label class="muted">Лимит:</label>
          <select id="fLimit">
            <option>25</option>
            <option selected>50</option>
            <option>100</option>
            <option>200</option>
          </select>

          <label class="muted">Поиск job_id:</label>
          <input id="fSearch" placeholder="начало job_id..." />

          <label class="muted">Интервал:</label>
          <select id="fInterval">
            <option value="2000">2s</option>
            <option value="3000" selected>3s</option>
            <option value="5000">5s</option>
            <option value="10000">10s</option>
          </select>

          <button class="primary" id="btnRefresh">Обновить</button>
          <button id="btnToggle">Пауза</button>
        </div>
        <div class="muted small" style="margin-top:10px">Подсказка: “Пауза” остановит запросы → перестанут сыпаться access-логи.</div>
      </div>
    </div>

    <div class="card">
      <table>
        <thead>
          <tr>
            <th style="width:190px">status</th>
            <th>job_id</th>
            <th style="width:220px">filename</th>
            <th style="width:220px">created</th>
            <th style="width:220px">finished</th>
            <th style="width:120px">action</th>
          </tr>
        </thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </div>

  <div class="modal" id="modal">
    <div class="card box pad">
      <div class="row" style="justify-content:space-between;align-items:flex-start">
        <div>
          <div class="muted small">Job detail</div>
          <div class="mono" id="mTitle" style="font-weight:900;margin-top:4px"></div>
        </div>
        <button class="danger" id="mClose">Закрыть</button>
      </div>
      <div class="pad" style="padding:12px 0 0 0">
        <pre id="mBody"></pre>
      </div>
    </div>
  </div>

<script>
  let timer = null;
  let auto = true;

  function fmt(ts){
    if(!ts) return "—";
    try { return new Date(ts).toLocaleString(); } catch(e){ return ts; }
  }

  function statusClass(s){
    return (s||"").toLowerCase();
  }

  async function fetchJson(url){
    const r = await fetch(url, {cache:"no-store"});
    if(!r.ok) throw new Error(url + " → " + r.status);
    return r.json();
  }

  async function refresh(){
    const status = document.getElementById("fStatus").value;
    const limit = document.getElementById("fLimit").value;
    const search = (document.getElementById("fSearch").value || "").trim();

    const stats = await fetchJson("/api/jobs/stats");
    document.getElementById("stQueued").innerText = stats.queued ?? 0;
    document.getElementById("stRunning").innerText = stats.running ?? 0;
    document.getElementById("stDone").innerText = stats.done ?? 0;
    document.getElementById("stError").innerText = stats.error ?? 0;
    document.getElementById("stTotal").innerText = stats.total ?? 0;

    let url = `/api/jobs?limit=${encodeURIComponent(limit)}`;
    if(status) url += `&status=${encodeURIComponent(status)}`;
    const jobs = await fetchJson(url);

    const tbody = document.getElementById("tbody");
    tbody.innerHTML = "";

    const filtered = search ? jobs.filter(j => (j.job_id||"").startsWith(search)) : jobs;

    for(const j of filtered){
      const tr = document.createElement("tr");
      const st = j.status || "";
      tr.innerHTML = `
        <td><span class="pill status ${statusClass(st)}">${st}</span></td>
        <td class="mono">${j.job_id || ""}</td>
        <td>${j.filename || "—"}</td>
        <td>${fmt(j.created_at)}</td>
        <td>${fmt(j.finished_at)}</td>
        <td><button class="primary small" data-id="${j.job_id}">Details</button></td>
      `;
      tbody.appendChild(tr);
    }

    document.getElementById("updated").innerText = "updated: " + new Date().toLocaleTimeString();
  }

  function start(){
    stop();
    const interval = parseInt(document.getElementById("fInterval").value, 10);
    if(auto){ timer = setInterval(refresh, interval); }
    document.getElementById("autoState").innerText = auto ? "ON" : "OFF";
  }

  function stop(){
    if(timer){ clearInterval(timer); timer = null; }
  }

  document.getElementById("btnRefresh").onclick = () => refresh();
  document.getElementById("btnToggle").onclick = () => {
    auto = !auto;
    document.getElementById("btnToggle").innerText = auto ? "Пауза" : "Продолжить";
    start();
  };

  document.getElementById("fStatus").onchange = () => refresh();
  document.getElementById("fLimit").onchange = () => refresh();
  document.getElementById("fInterval").onchange = () => start();

  document.getElementById("tbody").onclick = async (e) => {
    const btn = e.target.closest("button[data-id]");
    if(!btn) return;
    const id = btn.getAttribute("data-id");
    const data = await fetchJson("/api/jobs/" + id);
    document.getElementById("mTitle").innerText = id;
    document.getElementById("mBody").innerText = JSON.stringify(data, null, 2);
    document.getElementById("modal").classList.add("open");
  };

  document.getElementById("mClose").onclick = () => document.getElementById("modal").classList.remove("open");
  document.getElementById("modal").onclick = (e) => { if(e.target.id==="modal") e.target.classList.remove("open"); };

  refresh().then(start);
</script>
</body>
</html>
"""


# =========================================================
# STARTUP / SHUTDOWN
# =========================================================
@app.on_event("startup")
def startup() -> None:
    init_db()
    threading.Thread(target=worker, daemon=True).start()

    print("\n========================================")
    print("🚀 Server started")
    print(f"📊 Monitor: http://localhost:{PORT}/monitor")
    print("========================================\n")


@app.on_event("shutdown")
def shutdown() -> None:
    stop_event.set()