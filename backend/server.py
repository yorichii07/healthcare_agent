"""
Healthcare Planning Assistant Agent — FastAPI Backend
"""

import os
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from planner_agent import HEALTHCARE_KNOWLEDGE_BASE, PlannerAgent

app = FastAPI(title="Healthcare Planning Assistant API", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = PlannerAgent()


@app.on_event("startup")
def startup_diagnostics():
    # Log runtime details to make AI-mode initialization issues obvious.
    agent._refresh_llm_runtime()
    has_key = bool(os.getenv("GROQ_API_KEY", "").strip())
    print(f"[startup] python={sys.executable}")
    print(f"[startup] has_groq_key={has_key}")
    print(f"[startup] llm_enabled={agent.llm_enabled}")


class GoalRequest(BaseModel):
    goal: str
    mode: Literal["auto", "ai", "fallback"] = "auto"


@app.get("/")
def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/")
def api_root():
    return {"message": "Healthcare Planning Assistant Agent API is running"}


@app.post("/api/plan")
def create_plan(request: GoalRequest):
    if not request.goal.strip():
        raise HTTPException(status_code=400, detail="Goal cannot be empty")
    try:
        result = agent.create_plan(request.goal, mode=request.mode)
        return result
    except RuntimeError as e:
        error_msg = str(e)
        if (
            "AI mode requested, but Groq/LangChain is unavailable" in error_msg
            or "GROQ_API_KEY" in error_msg
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "AI runtime unavailable in the current server process. "
                    "Ensure the backend is started with the project's .venv Python, "
                    "and GROQ_API_KEY is set in .env."
                ),
            )
        if "model_decommissioned" in error_msg or "decommissioned" in error_msg:
            raise HTTPException(
                status_code=502,
                detail="Configured GROQ_MODEL is deprecated. Update GROQ_MODEL in .env to a supported value like llama-3.1-8b-instant.",
            )
        if "Error code: 503" in error_msg or "Service Unavailable" in error_msg:
            raise HTTPException(
                status_code=503,
                detail="AI provider is temporarily unavailable. Please retry in a few seconds or use Auto/Fallback mode.",
            )
        raise HTTPException(status_code=500, detail=f"AI planning failed: {error_msg}")


@app.get("/api/conditions")
def get_conditions():
    return {
        "conditions": [
            {"name": k, "description": v["description"], "task_count": len(v["tasks"])}
            for k, v in HEALTHCARE_KNOWLEDGE_BASE.items()
        ]
    }
