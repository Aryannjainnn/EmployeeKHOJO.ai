"""FastAPI entry. Serves the HTML frontend at `/` and the search API at `/search`."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from orchestrator import run as run_pipeline
from orchestrator.schemas import SearchRequest

load_dotenv()

BASE = Path(__file__).parent
FRONTEND_DIR = BASE / "frontend"

app = FastAPI(title="Intent-Aware Hybrid Retrieval")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.post("/search")
def search(req: SearchRequest) -> JSONResponse:
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query is required")
    try:
        result = run_pipeline(q)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"pipeline error: {e!s}")
    return JSONResponse(content=result)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
