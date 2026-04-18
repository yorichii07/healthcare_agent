"""
Healthcare Planning Assistant Agent
LangChain + Groq powered - Real LLM reasoning
"""

import json
import os
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv, find_dotenv

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    PromptTemplate = None

def _load_environment() -> None:
    # Load root and backend .env files so runtime works regardless of launch directory.
    backend_dir = Path(__file__).resolve().parent
    project_root = backend_dir.parent
    for env_path in (project_root / ".env", backend_dir / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=True)
    load_dotenv(find_dotenv(usecwd=True), override=True)


_load_environment()

# ─────────────────────────────────────────────
#  LLM SETUP
# ─────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


HEALTHCARE_KNOWLEDGE_BASE = {
    "diabetes": {
        "description": "Comprehensive Diabetes Management Plan",
        "keywords": ["diabetes", "blood sugar", "glucose", "hba1c", "metformin", "insulin"],
        "tasks": [
            {"description": "Consult an endocrinologist for diabetes assessment", "task_type": "consultation", "resource": "Endocrinologist", "dependencies": [], "priority": 3, "estimated_duration": "30 min"},
            {"description": "Check fasting blood glucose", "task_type": "lab_test", "resource": "Blood Glucose Fasting", "dependencies": [1], "priority": 3, "estimated_duration": "2 hours"},
            {"description": "Review long-term glycemic control", "task_type": "lab_test", "resource": "HbA1c", "dependencies": [1], "priority": 2, "estimated_duration": "6 hours"},
            {"description": "Start glucose-lowering medication if appropriate", "task_type": "medication", "resource": "Metformin", "dependencies": [2, 3], "priority": 3, "estimated_duration": "Ongoing"},
            {"description": "Schedule follow-up review", "task_type": "followup", "resource": "Endocrinologist", "dependencies": [4], "priority": 2, "estimated_duration": "7 days"},
        ],
    },
    "hypertension": {
        "description": "Blood Pressure Control and Cardiovascular Risk Reduction Plan",
        "keywords": ["hypertension", "high blood pressure", "bp", "pressure"],
        "tasks": [
            {"description": "Consult a cardiologist or general physician", "task_type": "consultation", "resource": "Cardiologist", "dependencies": [], "priority": 3, "estimated_duration": "30 min"},
            {"description": "Check ECG for cardiovascular screening", "task_type": "lab_test", "resource": "ECG", "dependencies": [1], "priority": 2, "estimated_duration": "30 minutes"},
            {"description": "Review lipid profile", "task_type": "lab_test", "resource": "Lipid Profile", "dependencies": [1], "priority": 2, "estimated_duration": "4 hours"},
            {"description": "Begin blood pressure medication if prescribed", "task_type": "medication", "resource": "Lisinopril", "dependencies": [2, 3], "priority": 3, "estimated_duration": "Ongoing"},
            {"description": "Schedule follow-up blood pressure review", "task_type": "followup", "resource": "Cardiologist", "dependencies": [4], "priority": 2, "estimated_duration": "7 days"},
        ],
    },
    "cardiac": {
        "description": "Cardiac Risk Assessment and Support Plan",
        "keywords": ["cardiac", "heart", "chest pain", "angina", "cardiology"],
        "tasks": [
            {"description": "Consult cardiology for symptom review", "task_type": "consultation", "resource": "Cardiologist", "dependencies": [], "priority": 3, "estimated_duration": "30 min"},
            {"description": "Perform ECG evaluation", "task_type": "lab_test", "resource": "ECG", "dependencies": [1], "priority": 3, "estimated_duration": "30 minutes"},
            {"description": "Review cholesterol and cardiovascular risk markers", "task_type": "lab_test", "resource": "Lipid Profile", "dependencies": [1], "priority": 2, "estimated_duration": "4 hours"},
            {"description": "Start antiplatelet support if appropriate", "task_type": "medication", "resource": "Aspirin", "dependencies": [2, 3], "priority": 3, "estimated_duration": "Ongoing"},
            {"description": "Plan follow-up cardiology review", "task_type": "followup", "resource": "Cardiologist", "dependencies": [4], "priority": 2, "estimated_duration": "7 days"},
        ],
    },
    "respiratory": {
        "description": "Respiratory Symptom Evaluation and Support Plan",
        "keywords": ["respiratory", "cough", "asthma", "breathing", "lungs", "pulmonary"],
        "tasks": [
            {"description": "Consult a pulmonologist for airway assessment", "task_type": "consultation", "resource": "Pulmonologist", "dependencies": [], "priority": 3, "estimated_duration": "30 min"},
            {"description": "Obtain a chest X-ray if symptoms warrant", "task_type": "lab_test", "resource": "Chest X-Ray", "dependencies": [1], "priority": 2, "estimated_duration": "2 days"},
            {"description": "Review bronchodilator therapy", "task_type": "medication", "resource": "Salbutamol", "dependencies": [1], "priority": 3, "estimated_duration": "Ongoing"},
            {"description": "Arrange follow-up respiratory review", "task_type": "followup", "resource": "Pulmonologist", "dependencies": [2, 3], "priority": 2, "estimated_duration": "7 days"},
        ],
    },
    "fever": {
        "description": "Fever and Infection Evaluation Plan",
        "keywords": ["fever", "infection", "temperature", "flu"],
        "tasks": [
            {"description": "Consult a general physician for symptom review", "task_type": "consultation", "resource": "General Physician", "dependencies": [], "priority": 3, "estimated_duration": "30 min"},
            {"description": "Check complete blood count", "task_type": "lab_test", "resource": "Complete Blood Count", "dependencies": [1], "priority": 3, "estimated_duration": "4 hours"},
            {"description": "Consider symptomatic treatment if recommended", "task_type": "medication", "resource": "Paracetamol", "dependencies": [1, 2], "priority": 2, "estimated_duration": "3 days"},
            {"description": "Plan follow-up if symptoms persist", "task_type": "followup", "resource": "General Physician", "dependencies": [3], "priority": 2, "estimated_duration": "3 days"},
        ],
    },
    "general checkup": {
        "description": "Preventive Health Checkup Plan",
        "keywords": ["checkup", "check-up", "annual", "routine", "screening"],
        "tasks": [
            {"description": "Consult a general physician for preventive review", "task_type": "consultation", "resource": "General Physician", "dependencies": [], "priority": 3, "estimated_duration": "30 min"},
            {"description": "Perform complete blood count screening", "task_type": "lab_test", "resource": "Complete Blood Count", "dependencies": [1], "priority": 2, "estimated_duration": "4 hours"},
            {"description": "Check lipid profile for risk assessment", "task_type": "lab_test", "resource": "Lipid Profile", "dependencies": [1], "priority": 2, "estimated_duration": "4 hours"},
            {"description": "Review routine follow-up and preventive counseling", "task_type": "followup", "resource": "General Physician", "dependencies": [2, 3], "priority": 1, "estimated_duration": "7 days"},
        ],
    },
}


def _normalize_condition(condition: str) -> str:
    return condition.strip().lower()


def _fallback_condition_for_goal(goal: str) -> tuple[str, str]:
    normalized_goal = goal.lower()
    for condition_name, info in HEALTHCARE_KNOWLEDGE_BASE.items():
        for keyword in info.get("keywords", []):
            if keyword in normalized_goal:
                return condition_name, info["description"]
    fallback = HEALTHCARE_KNOWLEDGE_BASE["general checkup"]
    return "general checkup", fallback["description"]


def _fallback_tasks_for_condition(condition: str) -> list:
    normalized_condition = _normalize_condition(condition)
    info = HEALTHCARE_KNOWLEDGE_BASE.get(normalized_condition) or HEALTHCARE_KNOWLEDGE_BASE["general checkup"]
    tasks = []
    for index, task_data in enumerate(info["tasks"], start=1):
        tasks.append(
            Task(
                task_id=index,
                description=task_data["description"],
                task_type=task_data["task_type"],
                resource=task_data["resource"],
                dependencies=task_data.get("dependencies", []),
                priority=task_data.get("priority", 2),
                estimated_duration=task_data.get("estimated_duration", "30 min"),
            )
        )
    return tasks


def _safe_json_loads(raw: str):
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


llm = None
if GROQ_API_KEY and ChatGroq is not None:
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.3,
    )


def _build_llm_client():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    if not api_key or ChatGroq is None:
        return None
    return ChatGroq(api_key=api_key, model_name=model, temperature=0.3)

# ─────────────────────────────────────────────
#  MOCK TOOL LAYER
# ─────────────────────────────────────────────

class DoctorAvailabilityAPI:
    DOCTORS = {
        "General Physician": ["Dr. Sharma", "Dr. Mehta"],
        "Cardiologist":      ["Dr. Patel"],
        "Neurologist":       ["Dr. Rao"],
        "Endocrinologist":   ["Dr. Gupta"],
        "Pulmonologist":     ["Dr. Joshi"],
        "Orthopedist":       ["Dr. Verma"],
        "Dermatologist":     ["Dr. Kapoor"],
        "Psychiatrist":      ["Dr. Nair"],
    }

    def check_availability(self, specialty: str) -> dict:
        time.sleep(0.1)
        doctors = self.DOCTORS.get(specialty, [])
        if doctors:
            doctor = random.choice(doctors)
            slot = (datetime.now() + timedelta(hours=random.randint(2, 48))).strftime("%Y-%m-%d %H:%M")
            return {"available": True, "doctor": doctor, "next_slot": slot, "specialty": specialty}
        return {"available": False, "doctor": None, "next_slot": None, "specialty": specialty}


class MedicineDatabaseAPI:
    MEDICINES = {
        "Paracetamol":  {"stock": True,  "alternatives": []},
        "Metformin":    {"stock": True,  "alternatives": ["Glucophage"]},
        "Lisinopril":   {"stock": False, "alternatives": ["Enalapril", "Ramipril"]},
        "Atorvastatin": {"stock": True,  "alternatives": []},
        "Amoxicillin":  {"stock": True,  "alternatives": ["Azithromycin"]},
        "Aspirin":      {"stock": True,  "alternatives": []},
        "Insulin":      {"stock": True,  "alternatives": ["Glargine"]},
        "Salbutamol":   {"stock": False, "alternatives": ["Formoterol"]},
        "Omeprazole":   {"stock": True,  "alternatives": []},
        "Cetirizine":   {"stock": True,  "alternatives": ["Loratadine"]},
    }

    def check_stock(self, medicine: str) -> dict:
        time.sleep(0.1)
        info = self.MEDICINES.get(medicine, {"stock": True, "alternatives": []})
        return {"medicine": medicine, "in_stock": info["stock"], "alternatives": info["alternatives"]}


class LabTestAPI:
    TESTS = {
        "Complete Blood Count":  {"available": True,  "turnaround": "4 hours"},
        "Blood Glucose Fasting": {"available": True,  "turnaround": "2 hours"},
        "HbA1c":                 {"available": True,  "turnaround": "6 hours"},
        "Lipid Profile":         {"available": True,  "turnaround": "4 hours"},
        "ECG":                   {"available": True,  "turnaround": "30 minutes"},
        "Chest X-Ray":           {"available": False, "turnaround": None},
        "MRI Brain":             {"available": True,  "turnaround": "2 days"},
        "Thyroid Profile":       {"available": True,  "turnaround": "6 hours"},
        "Urine Routine":         {"available": True,  "turnaround": "2 hours"},
        "Liver Function Test":   {"available": True,  "turnaround": "5 hours"},
        "Kidney Function Test":  {"available": True,  "turnaround": "5 hours"},
    }

    def check_test_availability(self, test_name: str) -> dict:
        time.sleep(0.1)
        info = self.TESTS.get(test_name, {"available": True, "turnaround": "1 day"})
        return {"test": test_name, "available": info["available"], "turnaround": info["turnaround"]}


class ToolManager:
    def __init__(self):
        self.doctor_api   = DoctorAvailabilityAPI()
        self.medicine_api = MedicineDatabaseAPI()
        self.lab_api      = LabTestAPI()

    def call_tool(self, tool_name: str, params: dict) -> dict:
        if tool_name == "check_doctor":
            return self.doctor_api.check_availability(params.get("specialty", "General Physician"))
        elif tool_name == "check_medicine":
            return self.medicine_api.check_stock(params.get("medicine", ""))
        elif tool_name == "check_lab":
            return self.lab_api.check_test_availability(params.get("test", ""))
        return {"error": f"Unknown tool: {tool_name}"}


# ─────────────────────────────────────────────
#  DATA MODELS
# ─────────────────────────────────────────────

class Task:
    def __init__(self, task_id, description, task_type, resource,
                 dependencies=None, priority=1, estimated_duration="30 min"):
        self.id                 = task_id
        self.description        = description
        self.task_type          = task_type
        self.resource           = resource
        self.dependencies       = dependencies or []
        self.priority           = priority
        self.estimated_duration = estimated_duration
        self.status             = "pending"
        self.validation_result  = None
        self.scheduled_time     = None
        self.notes              = ""

    def to_dict(self):
        return {
            "id":                 self.id,
            "description":        self.description,
            "task_type":          self.task_type,
            "resource":           self.resource,
            "dependencies":       self.dependencies,
            "priority":           self.priority,
            "estimated_duration": self.estimated_duration,
            "status":             self.status,
            "validation_result":  self.validation_result,
            "scheduled_time":     self.scheduled_time,
            "notes":              self.notes,
        }


class ExecutionPlan:
    def __init__(self, goal, tasks, timeline, summary):
        self.goal       = goal
        self.tasks      = tasks
        self.timeline   = timeline
        self.summary    = summary
        self.created_at = datetime.now().isoformat()

    def to_dict(self):
        return {
            "goal":       self.goal,
            "tasks":      [t.to_dict() for t in self.tasks],
            "timeline":   self.timeline,
            "summary":    self.summary,
            "created_at": self.created_at,
        }


class MemoryStore:
    def __init__(self):
        self._store = []

    def add(self, entry: dict):
        entry["timestamp"] = datetime.now().isoformat()
        self._store.append(entry)

    def get_all(self):
        return self._store


class Scheduler:
    def optimize_tasks(self, tasks):
        task_map = {t.id: t for t in tasks}
        visited, result = set(), []

        def dfs(task_id):
            if task_id in visited:
                return
            visited.add(task_id)
            for dep_id in task_map[task_id].dependencies:
                if dep_id in task_map:
                    dfs(dep_id)
            result.append(task_map[task_id])

        for t in tasks:
            dfs(t.id)

        result.sort(key=lambda x: -x.priority)
        return result

    def generate_timeline(self, tasks):
        timeline, current_time = [], datetime.now()
        for i, task in enumerate(tasks):
            task.scheduled_time = current_time.strftime("%Y-%m-%d %H:%M")
            timeline.append({
                "step":           i + 1,
                "task_id":        task.id,
                "description":    task.description,
                "type":           task.task_type,
                "scheduled_time": task.scheduled_time,
                "duration":       task.estimated_duration,
                "status":         task.status,
            })
            hours = 1 if "min" in task.estimated_duration.lower() else 4
            current_time += timedelta(hours=hours)
        return timeline


# ─────────────────────────────────────────────
#  LANGCHAIN PROMPTS
# ─────────────────────────────────────────────

if PromptTemplate is not None:
    CONDITION_PROMPT = PromptTemplate(
        input_variables=["goal"],
        template="""
You are a healthcare AI assistant. A user has given this goal:
"{goal}"

Identify the primary medical condition and return a JSON object with:
- "condition": the condition in lowercase (e.g. diabetes, fever, hypertension, cardiac, respiratory)
- "description": a short professional plan title (e.g. "Comprehensive Diabetes Management Plan")

Respond ONLY with valid JSON. No explanation. No markdown. No extra text.
Example: {{"condition": "diabetes", "description": "Comprehensive Diabetes Management Plan"}}
"""
    )

    TASK_DECOMPOSITION_PROMPT = PromptTemplate(
        input_variables=["goal", "condition"],
        template="""
You are a healthcare planning AI. Generate a structured treatment plan for:
Goal: "{goal}"
Condition: "{condition}"

Return a JSON array of tasks. Each task must have:
- "id": integer starting from 1
- "description": clear task description
- "task_type": one of ["consultation", "lab_test", "medication", "followup"]
- "resource": doctor specialty, medicine name, or lab test name (be specific)
- "dependencies": list of task ids that must complete before this one ([] if none)
- "priority": integer 1-3 (3=high, 2=medium, 1=low)
- "estimated_duration": string like "30 min", "2 hours", "Ongoing", "7 days"

Rules:
- consultation: resource = doctor specialty (e.g. "General Physician", "Cardiologist", "Endocrinologist")
- lab_test: resource = exact test name (e.g. "Complete Blood Count", "ECG", "HbA1c")
- medication: resource = medicine name (e.g. "Metformin", "Aspirin", "Paracetamol")
- followup: resource = doctor specialty
- Generate 5-8 tasks total
- Consultation must come first (no dependencies), tests before medications

Respond ONLY with a valid JSON array. No explanation. No markdown. No extra text.
"""
    )
else:
    CONDITION_PROMPT = None
    TASK_DECOMPOSITION_PROMPT = None


# ─────────────────────────────────────────────
#  PLANNER AGENT
# ─────────────────────────────────────────────

class PlannerAgent:
    def __init__(self):
        self.tool_manager  = ToolManager()
        self.scheduler     = Scheduler()
        self.memory        = MemoryStore()
        self.reasoning_log = []
        self.llm_enabled = False
        self.condition_chain = None
        self.task_chain = None
        self._refresh_llm_runtime()

    def _refresh_llm_runtime(self):
        _load_environment()
        llm_client = _build_llm_client()
        self.llm_enabled = (
            llm_client is not None
            and CONDITION_PROMPT is not None
            and TASK_DECOMPOSITION_PROMPT is not None
        )
        self.condition_chain = llm_client if self.llm_enabled else None
        self.task_chain = llm_client if self.llm_enabled else None

    @staticmethod
    def _invoke_llm_with_prompt(llm_client, prompt_template, **kwargs) -> str:
        prompt_text = prompt_template.format(**kwargs)
        response = llm_client.invoke(prompt_text)
        return response.content if hasattr(response, "content") else str(response)

    def _log(self, msg: str):
        self.reasoning_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def understand_goal(self, goal: str, mode: str = "auto") -> tuple:
        if mode not in {"auto", "ai", "fallback"}:
            raise ValueError("mode must be one of: auto, ai, fallback")

        if mode != "fallback":
            if self.llm_enabled:
                try:
                    self._log(f"Sending goal to Groq LLM: '{goal}'")
                    raw = self._invoke_llm_with_prompt(self.condition_chain, CONDITION_PROMPT, goal=goal)
                    parsed = _safe_json_loads(raw)
                    condition   = parsed.get("condition", "general checkup")
                    description = parsed.get("description", "Healthcare Plan")
                    self._log(f"LLM detected: '{condition}' → {description}")
                    self.memory.add({"type": "goal_understood", "condition": condition})
                    return condition, description
                except Exception as exc:
                    if mode == "ai":
                        raise RuntimeError(f"AI mode failed while understanding goal: {exc}") from exc
                    self._log(f"LLM goal understanding failed, using fallback rules: {exc}")
            elif mode == "ai":
                raise RuntimeError("AI mode requested, but Groq/LangChain is unavailable")

        condition, description = _fallback_condition_for_goal(goal)
        self._log(f"Fallback detected: '{condition}' → {description}")
        self.memory.add({"type": "goal_understood", "condition": condition})
        return condition, description

    def decompose_tasks(self, goal: str, condition: str, mode: str = "auto") -> list:
        if mode not in {"auto", "ai", "fallback"}:
            raise ValueError("mode must be one of: auto, ai, fallback")

        if mode != "fallback":
            if self.llm_enabled:
                try:
                    self._log("LLM decomposing goal into tasks...")
                    raw = self._invoke_llm_with_prompt(
                        self.task_chain,
                        TASK_DECOMPOSITION_PROMPT,
                        goal=goal,
                        condition=condition,
                    )
                    task_data = _safe_json_loads(raw)
                    tasks = []
                    for t in task_data:
                        task = Task(
                            task_id            = t["id"],
                            description        = t["description"],
                            task_type          = t["task_type"],
                            resource           = t["resource"],
                            dependencies       = t.get("dependencies", []),
                            priority           = t.get("priority", 2),
                            estimated_duration = t.get("estimated_duration", "30 min"),
                        )
                        tasks.append(task)
                        self._log(f"  Task {task.id}: {task.description} [{task.task_type}]")
                    self.memory.add({"type": "tasks_decomposed", "count": len(tasks)})
                    return tasks
                except Exception as exc:
                    if mode == "ai":
                        raise RuntimeError(f"AI mode failed while decomposing tasks: {exc}") from exc
                    self._log(f"LLM task decomposition failed, using fallback rules: {exc}")
            elif mode == "ai":
                raise RuntimeError("AI mode requested, but Groq/LangChain is unavailable")

        tasks = _fallback_tasks_for_condition(condition)
        for task in tasks:
            self._log(f"  Task {task.id}: {task.description} [{task.task_type}]")
        self.memory.add({"type": "tasks_decomposed", "count": len(tasks)})
        return tasks

    def validate_resources(self, tasks: list) -> list:
        self._log("Validating resources via mock tool APIs...")
        for task in tasks:
            if task.task_type == "consultation":
                result = self.tool_manager.call_tool("check_doctor", {"specialty": task.resource})
                task.validation_result = result
                if result["available"]:
                    task.status = "validated"
                    task.notes  = f"Assigned to {result['doctor']} | Slot: {result['next_slot']}"
                    self._log(f"  ✓ Task {task.id}: {result['doctor']} available")
                else:
                    task.status = "unavailable"
                    task.notes  = f"No {task.resource} available. Consider teleconsultation."
                    self._log(f"  ✗ Task {task.id}: {task.resource} unavailable")

            elif task.task_type == "lab_test":
                result = self.tool_manager.call_tool("check_lab", {"test": task.resource})
                task.validation_result = result
                if result["available"]:
                    task.status = "validated"
                    task.notes  = f"Turnaround: {result['turnaround']}"
                    self._log(f"  ✓ Task {task.id}: Lab available, TAT={result['turnaround']}")
                else:
                    task.status = "unavailable"
                    task.notes  = "Lab test unavailable. Consider alternate facility."
                    self._log(f"  ✗ Task {task.id}: Lab unavailable")

            elif task.task_type == "medication":
                result = self.tool_manager.call_tool("check_medicine", {"medicine": task.resource})
                task.validation_result = result
                if result["in_stock"]:
                    task.status = "validated"
                    task.notes  = f"{task.resource} in stock."
                    self._log(f"  ✓ Task {task.id}: {task.resource} in stock")
                else:
                    alts = result["alternatives"]
                    task.status = "alternative_found" if alts else "unavailable"
                    task.notes  = (f"Out of stock. Alternatives: {', '.join(alts)}"
                                   if alts else "Out of stock. No alternatives.")
                    self._log(f"  ⚠ Task {task.id}: {task.resource} out of stock")

            elif task.task_type == "followup":
                task.status = "scheduled"
                task.notes  = "Follow-up to be confirmed after primary treatment."
                self._log(f"  ✓ Task {task.id}: Follow-up scheduled")

        self.memory.add({"type": "resources_validated"})
        return tasks

    def schedule_and_optimise(self, tasks: list) -> tuple:
        self._log("Resolving dependencies & optimising task order...")
        ordered  = self.scheduler.optimize_tasks(tasks)
        timeline = self.scheduler.generate_timeline(ordered)
        self._log(f"Timeline generated with {len(timeline)} steps")
        return ordered, timeline

    def build_summary(self, condition: str, tasks: list) -> str:
        nVal = sum(1 for t in tasks if t.status == "validated")
        nUna = sum(1 for t in tasks if t.status == "unavailable")
        nAlt = sum(1 for t in tasks if t.status == "alternative_found")
        return (
            f"Healthcare plan for '{condition}' with {len(tasks)} tasks — "
            f"{nVal} validated, {nUna} unavailable, {nAlt} with alternatives."
        )

    def create_plan(self, goal: str, mode: str = "auto") -> dict:
        self.reasoning_log = []
        self._refresh_llm_runtime()
        self._log(f"=== Planner Agent Started (mode={mode}) ===")

        condition, description = self.understand_goal(goal, mode=mode)
        tasks                  = self.decompose_tasks(goal, condition, mode=mode)
        tasks                  = self.validate_resources(tasks)
        ordered, timeline      = self.schedule_and_optimise(tasks)
        summary                = self.build_summary(condition, ordered)

        plan = ExecutionPlan(goal=goal, tasks=ordered, timeline=timeline, summary=summary)
        self._log("=== Plan Complete ===")

        return {
            "plan":          plan.to_dict(),
            "reasoning_log": self.reasoning_log,
            "condition":     condition,
            "description":   description,
        }


if __name__ == "__main__":
    agent  = PlannerAgent()
    result = agent.create_plan("Treatment plan for diabetes management")
    print(json.dumps(result, indent=2))
