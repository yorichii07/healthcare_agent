"""
Healthcare Planning Assistant Agent — FastAPI Backend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from planner_agent import PlannerAgent

app = FastAPI(title="Healthcare Planning Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = PlannerAgent()


class GoalRequest(BaseModel):
    goal: str


@app.get("/")
def root():
    return {"message": "Healthcare Planning Assistant Agent API is running"}


@app.post("/plan")
def create_plan(request: GoalRequest):
    if not request.goal.strip():
        raise HTTPException(status_code=400, detail="Goal cannot be empty")
    result = agent.create_plan(request.goal)
    return result


@app.get("/conditions")
def get_conditions():
    from planner_agent import HEALTHCARE_KNOWLEDGE_BASE
    return {
        "conditions": [
            {"name": k, "description": v["description"], "task_count": len(v["tasks"])}
            for k, v in HEALTHCARE_KNOWLEDGE_BASE.items()
        ]
    }
