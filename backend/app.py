"""
FastAPI server for the Shorts Generator.

Endpoints:
  POST /api/process        {url: str}         -> {job_id}
  GET  /api/status/{job_id}                    -> {status, message, error, ready}
  GET  /api/download/{job_id}                  -> the rendered .mp4
  GET  /                                        -> mobile frontend (index.html)
"""

import threading
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import run_pipeline

app = FastAPI(title="Shorts Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: dict[str, dict] = {}


class ProcessRequest(BaseModel):
    url: str


@app.post("/api/process")
def process(req: ProcessRequest):
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "queued", "message": "Queued...", "ready": False, "error": None}

    def worker():
        try:
            def progress(msg):
                JOBS[job_id]["message"] = msg
                JOBS[job_id]["status"] = "running"

            short_path, highlight = run_pipeline(req.url, job_id, progress_cb=progress)
            JOBS[job_id].update(
                status="done",
                message="Ready!",
                ready=True,
                file_path=str(short_path),
                highlight=highlight,
            )
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            JOBS[job_id].update(status="error", error=str(e), message="Failed", ready=False)

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "status": job["status"],
        "message": job["message"],
        "ready": job["ready"],
        "error": job.get("error"),
        "highlight": job.get("highlight"),
    }


@app.get("/api/download/{job_id}")
def download(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.get("ready"):
        raise HTTPException(404, "File not ready")
    return FileResponse(job["file_path"], media_type="video/mp4", filename="short.mp4")


frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
