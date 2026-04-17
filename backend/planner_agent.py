"""
Healthcare Planning Assistant Agent
Core Planner Agent with multi-step reasoning loop
"""

import json
import time
import random
from datetime import datetime, timedelta
from typing import Optional


# ─────────────────────────────────────────────
#  MOCK TOOL LAYER  (simulates real-world APIs)
# ─────────────────────────────────────────────

class DoctorAvailabilityAPI:
    """Mock Doctor Availability Service"""
    DOCTORS = {
        "General Physician": ["Dr. Sharma", "Dr. Mehta"],
        "Cardiologist":      ["Dr. Patel"],
        "Neurologist":       ["Dr. Rao"],
        "Endocrinologist":   ["Dr. Gupta"],
        "Pulmonologist":     ["Dr. Joshi"],
        "Orthopedist":       ["Dr. Verma"],
    }

    def check_availability(self, specialty: str) -> dict:
        time.sleep(0.1)   # simulate network latency
        doctors = self.DOCTORS.get(specialty, [])
        if doctors:
            doctor = random.choice(doctors)
            slot = (datetime.now() + timedelta(hours=random.randint(2, 48))).strftime("%Y-%m-%d %H:%M")
            return {"available": True, "doctor": doctor, "next_slot": slot, "specialty": specialty}
        return {"available": False, "doctor": None, "next_slot": None, "specialty": specialty}


class MedicineDatabaseAPI:
    """Mock Medicine Stock / Prescription Service"""
    MEDICINES = {
        "Paracetamol":   {"stock": True,  "alternatives": []},
        "Metformin":     {"stock": True,  "alternatives": ["Glucophage"]},
        "Lisinopril":    {"stock": False, "alternatives": ["Enalapril", "Ramipril"]},
        "Atorvastatin":  {"stock": True,  "alternatives": []},
        "Amoxicillin":   {"stock": True,  "alternatives": ["Azithromycin"]},
        "Aspirin":       {"stock": True,  "alternatives": []},
        "Insulin":       {"stock": True,  "alternatives": ["Glargine"]},
        "Salbutamol":    {"stock": False, "alternatives": ["Formoterol"]},
    }

    def check_stock(self, medicine: str) -> dict:
        time.sleep(0.1)
        info = self.MEDICINES.get(medicine, {"stock": True, "alternatives": []})
        return {"medicine": medicine, "in_stock": info["stock"], "alternatives": info["alternatives"]}


class LabTestAPI:
    """Mock Laboratory Test Booking Service"""
    TESTS = {
        "Complete Blood Count":     {"available": True,  "turnaround": "4 hours"},
        "Blood Glucose Fasting":    {"available": True,  "turnaround": "2 hours"},
        "HbA1c":                    {"available": True,  "turnaround": "6 hours"},
        "Lipid Profile":            {"available": True,  "turnaround": "4 hours"},
        "ECG":                      {"available": True,  "turnaround": "30 minutes"},
        "Chest X-Ray":              {"available": False, "turnaround": None},
        "MRI Brain":                {"available": True,  "turnaround": "2 days"},
        "Thyroid Profile":          {"available": True,  "turnaround": "6 hours"},
        "Urine Routine":            {"available": True,  "turnaround": "2 hours"},
    }

    def check_test_availability(self, test_name: str) -> dict:
        time.sleep(0.1)
        info = self.TESTS.get(test_name, {"available": True, "turnaround": "1 day"})
        return {"test": test_name, "available": info["available"], "turnaround": info["turnaround"]}


# ─────────────────────────────────────────────
#  TOOL MANAGER
# ─────────────────────────────────────────────

class ToolManager:
    """Routes tool calls to the correct mock API"""
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
#  TASK & EXECUTION PLAN DATA MODELS
# ─────────────────────────────────────────────

class Task:
    def __init__(self, task_id: int, description: str, task_type: str,
                 resource: str, dependencies: list[int] = None,
                 priority: int = 1, estimated_duration: str = "30 min"):
        self.id                 = task_id
        self.description        = description
        self.task_type          = task_type          # "consultation"|"lab_test"|"medication"|"followup"
        self.resource           = resource
        self.dependencies       = dependencies or []
        self.priority           = priority
        self.estimated_duration = estimated_duration
        self.status             = "pending"
        self.validation_result  = None
        self.scheduled_time     = None
        self.notes              = ""

    def to_dict(self) -> dict:
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
    def __init__(self, goal: str, tasks: list[Task], timeline: list[dict], summary: str):
        self.goal      = goal
        self.tasks     = tasks
        self.timeline  = timeline
        self.summary   = summary
        self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "goal":       self.goal,
            "tasks":      [t.to_dict() for t in self.tasks],
            "timeline":   self.timeline,
            "summary":    self.summary,
            "created_at": self.created_at,
        }


# ─────────────────────────────────────────────
#  TASK DECOMPOSITION  (LLM-style knowledge base)
# ─────────────────────────────────────────────

HEALTHCARE_KNOWLEDGE_BASE = {
    "diabetes": {
        "tasks": [
            Task(1, "Initial Consultation with Endocrinologist",    "consultation", "Endocrinologist",   [],    3, "45 min"),
            Task(2, "Fasting Blood Glucose Test",                   "lab_test",     "Blood Glucose Fasting", [1], 3, "2 hours"),
            Task(3, "HbA1c Test",                                   "lab_test",     "HbA1c",             [1],   3, "6 hours"),
            Task(4, "Lipid Profile Test",                           "lab_test",     "Lipid Profile",     [1],   2, "4 hours"),
            Task(5, "Prescribe Metformin (if Type 2 confirmed)",    "medication",   "Metformin",         [2,3], 3, "Ongoing"),
            Task(6, "Dietary & Lifestyle Counselling",              "consultation", "General Physician", [3],   2, "60 min"),
            Task(7, "Follow-up with Endocrinologist (4 weeks)",     "followup",     "Endocrinologist",   [5,6], 2, "30 min"),
        ],
        "description": "Comprehensive Diabetes Management Plan"
    },
    "fever": {
        "tasks": [
            Task(1, "Initial Symptom Assessment",                   "consultation", "General Physician", [],    3, "20 min"),
            Task(2, "Complete Blood Count (CBC) Test",              "lab_test",     "Complete Blood Count", [1], 3, "4 hours"),
            Task(3, "Urine Routine Examination",                    "lab_test",     "Urine Routine",     [1],   2, "2 hours"),
            Task(4, "Prescribe Paracetamol for fever management",   "medication",   "Paracetamol",       [1],   3, "As needed"),
            Task(5, "Prescribe Amoxicillin if bacterial infection", "medication",   "Amoxicillin",       [2],   2, "7 days"),
            Task(6, "Follow-up if fever persists > 3 days",         "followup",     "General Physician", [2,3], 1, "20 min"),
        ],
        "description": "Acute Fever Diagnosis & Treatment Plan"
    },
    "hypertension": {
        "tasks": [
            Task(1, "BP Measurement & Initial Assessment",          "consultation", "General Physician", [],    3, "30 min"),
            Task(2, "ECG",                                          "lab_test",     "ECG",               [1],   3, "30 minutes"),
            Task(3, "Lipid Profile Test",                           "lab_test",     "Lipid Profile",     [1],   2, "4 hours"),
            Task(4, "Prescribe Lisinopril (ACE Inhibitor)",         "medication",   "Lisinopril",        [1],   3, "Ongoing"),
            Task(5, "Prescribe Aspirin (if cardiac risk)",          "medication",   "Aspirin",           [2],   2, "Ongoing"),
            Task(6, "Lifestyle Modification Counselling",           "consultation", "General Physician", [1],   2, "45 min"),
            Task(7, "Cardiology Referral if Stage 2",               "consultation", "Cardiologist",      [2,3], 2, "45 min"),
            Task(8, "Follow-up BP Check (2 weeks)",                 "followup",     "General Physician", [4,6], 2, "20 min"),
        ],
        "description": "Hypertension Assessment & Management Plan"
    },
    "cardiac": {
        "tasks": [
            Task(1, "Emergency Cardiac Assessment",                 "consultation", "Cardiologist",      [],    3, "60 min"),
            Task(2, "ECG",                                          "lab_test",     "ECG",               [1],   3, "30 minutes"),
            Task(3, "Complete Blood Count",                         "lab_test",     "Complete Blood Count", [1], 3, "4 hours"),
            Task(4, "Lipid Profile",                                "lab_test",     "Lipid Profile",     [1],   3, "4 hours"),
            Task(5, "Prescribe Atorvastatin",                       "medication",   "Atorvastatin",      [4],   3, "Ongoing"),
            Task(6, "Prescribe Aspirin",                            "medication",   "Aspirin",           [2],   3, "Ongoing"),
            Task(7, "Cardiac Rehabilitation Program",               "consultation", "Cardiologist",      [2,3], 2, "90 min"),
            Task(8, "Follow-up Cardiology (1 week)",                "followup",     "Cardiologist",      [5,6], 3, "30 min"),
        ],
        "description": "Cardiac Care & Treatment Plan"
    },
    "respiratory": {
        "tasks": [
            Task(1, "Respiratory Assessment",                       "consultation", "Pulmonologist",     [],    3, "45 min"),
            Task(2, "Chest X-Ray",                                  "lab_test",     "Chest X-Ray",       [1],   3, "30 min"),
            Task(3, "Complete Blood Count",                         "lab_test",     "Complete Blood Count", [1], 2, "4 hours"),
            Task(4, "Prescribe Salbutamol Inhaler",                 "medication",   "Salbutamol",        [1],   3, "As needed"),
            Task(5, "Breathing Exercise & Physiotherapy",           "consultation", "Pulmonologist",     [2],   2, "60 min"),
            Task(6, "Follow-up in 2 weeks",                         "followup",     "Pulmonologist",     [4,5], 2, "30 min"),
        ],
        "description": "Respiratory Condition Management Plan"
    },
    "general checkup": {
        "tasks": [
            Task(1, "Comprehensive Physical Examination",           "consultation", "General Physician", [],    2, "60 min"),
            Task(2, "Complete Blood Count",                         "lab_test",     "Complete Blood Count", [1], 2, "4 hours"),
            Task(3, "Blood Glucose Fasting",                        "lab_test",     "Blood Glucose Fasting", [1], 2, "2 hours"),
            Task(4, "Lipid Profile",                                "lab_test",     "Lipid Profile",     [1],   2, "4 hours"),
            Task(5, "Thyroid Profile",                              "lab_test",     "Thyroid Profile",   [1],   1, "6 hours"),
            Task(6, "Review Results & Recommendations",             "consultation", "General Physician", [2,3,4,5], 2, "30 min"),
        ],
        "description": "Annual Health Checkup Plan"
    },
}


def detect_condition(goal: str) -> tuple[str, dict]:
    """Detect the healthcare condition from the user's goal text."""
    goal_lower = goal.lower()
    for condition, data in HEALTHCARE_KNOWLEDGE_BASE.items():
        if condition in goal_lower:
            return condition, data
    # fallback
    return "general checkup", HEALTHCARE_KNOWLEDGE_BASE["general checkup"]


# ─────────────────────────────────────────────
#  MEMORY STORE
# ─────────────────────────────────────────────

class MemoryStore:
    def __init__(self):
        self._store: list[dict] = []

    def add(self, entry: dict):
        entry["timestamp"] = datetime.now().isoformat()
        self._store.append(entry)

    def get_all(self) -> list[dict]:
        return self._store

    def get_last(self, n: int = 5) -> list[dict]:
        return self._store[-n:]


# ─────────────────────────────────────────────
#  SCHEDULER
# ─────────────────────────────────────────────

class Scheduler:
    """Topological sort + timeline generation"""

    def optimize_tasks(self, tasks: list[Task]) -> list[Task]:
        """Topological sort respecting dependencies."""
        task_map = {t.id: t for t in tasks}
        visited, result = set(), []

        def dfs(task_id):
            if task_id in visited:
                return
            visited.add(task_id)
            for dep_id in task_map[task_id].dependencies:
                dfs(dep_id)
            result.append(task_map[task_id])

        for t in tasks:
            dfs(t.id)

        # secondary sort: by priority (descending)
        result.sort(key=lambda x: -x.priority)
        return result

    def generate_timeline(self, tasks: list[Task]) -> list[dict]:
        """Assign scheduled times and build a human-readable timeline."""
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
            # advance clock
            hours = 1 if "min" in task.estimated_duration.lower() else 4
            current_time += timedelta(hours=hours)
        return timeline


# ─────────────────────────────────────────────
#  PLANNER AGENT  (core reasoning loop)
# ─────────────────────────────────────────────

class PlannerAgent:
    """
    The core agent.  Reasoning loop:
      1. Understand goal
      2. Decompose into tasks
      3. Validate resources via tools
      4. Resolve dependencies
      5. Optimise order & generate timeline
      6. Return execution plan
    """

    def __init__(self):
        self.tool_manager = ToolManager()
        self.scheduler    = Scheduler()
        self.memory       = MemoryStore()
        self.reasoning_log: list[str] = []

    def _log(self, msg: str):
        self.reasoning_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    # ── Step 1 ──────────────────────────────
    def understand_goal(self, goal: str) -> tuple[str, dict]:
        self._log(f"Understanding goal: '{goal}'")
        condition, data = detect_condition(goal)
        self._log(f"Detected condition: '{condition}' → {data['description']}")
        self.memory.add({"type": "goal_understood", "condition": condition, "goal": goal})
        return condition, data

    # ── Step 2 ──────────────────────────────
    def decompose_tasks(self, data: dict) -> list[Task]:
        import copy
        tasks = [copy.deepcopy(t) for t in data["tasks"]]
        self._log(f"Decomposed into {len(tasks)} tasks")
        for t in tasks:
            self._log(f"  Task {t.id}: {t.description} (deps={t.dependencies})")
        self.memory.add({"type": "tasks_decomposed", "count": len(tasks)})
        return tasks

    # ── Step 3 ──────────────────────────────
    def validate_resources(self, tasks: list[Task]) -> list[Task]:
        self._log("Validating resources via tool layer …")
        for task in tasks:
            if task.task_type == "consultation":
                result = self.tool_manager.call_tool("check_doctor", {"specialty": task.resource})
                task.validation_result = result
                if result["available"]:
                    task.status = "validated"
                    task.notes  = f"Assigned to {result['doctor']} | Slot: {result['next_slot']}"
                    self._log(f"  ✓ Task {task.id}: {result['doctor']} available at {result['next_slot']}")
                else:
                    task.status = "unavailable"
                    task.notes  = f"No {task.resource} available. Consider teleconsultation."
                    self._log(f"  ✗ Task {task.id}: No {task.resource} available")

            elif task.task_type == "lab_test":
                result = self.tool_manager.call_tool("check_lab", {"test": task.resource})
                task.validation_result = result
                if result["available"]:
                    task.status = "validated"
                    task.notes  = f"Turnaround: {result['turnaround']}"
                    self._log(f"  ✓ Task {task.id}: Lab test available, TAT={result['turnaround']}")
                else:
                    task.status = "unavailable"
                    task.notes  = "Lab test currently unavailable. Consider alternate facility."
                    self._log(f"  ✗ Task {task.id}: Lab test unavailable")

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
                                   if alts else "Out of stock. No alternatives found.")
                    self._log(f"  ⚠ Task {task.id}: {task.resource} out of stock → {task.notes}")

            elif task.task_type == "followup":
                task.status = "scheduled"
                task.notes  = "Follow-up appointment to be confirmed after primary treatment."
                self._log(f"  ✓ Task {task.id}: Follow-up scheduled")

        self.memory.add({"type": "resources_validated",
                         "validated": sum(1 for t in tasks if t.status == "validated")})
        return tasks

    # ── Step 4 + 5 ───────────────────────────
    def schedule_and_optimise(self, tasks: list[Task]) -> tuple[list[Task], list[dict]]:
        self._log("Resolving dependencies & optimising task order …")
        ordered = self.scheduler.optimize_tasks(tasks)
        timeline = self.scheduler.generate_timeline(ordered)
        self._log(f"Timeline generated with {len(timeline)} steps")
        self.memory.add({"type": "plan_generated", "steps": len(timeline)})
        return ordered, timeline

    # ── Step 6 ───────────────────────────────
    def build_summary(self, condition: str, tasks: list[Task]) -> str:
        validated = [t for t in tasks if t.status == "validated"]
        unavail   = [t for t in tasks if t.status == "unavailable"]
        alt       = [t for t in tasks if t.status == "alternative_found"]
        return (
            f"Healthcare plan for '{condition}' generated with {len(tasks)} tasks. "
            f"{len(validated)} validated, {len(unavail)} unavailable, "
            f"{len(alt)} with alternatives suggested."
        )

    # ── Main entry point ──────────────────────
    def create_plan(self, goal: str) -> dict:
        self.reasoning_log = []
        self._log("=== Planner Agent Started ===")

        condition, data   = self.understand_goal(goal)
        tasks             = self.decompose_tasks(data)
        tasks             = self.validate_resources(tasks)
        ordered, timeline = self.schedule_and_optimise(tasks)
        summary           = self.build_summary(condition, ordered)

        plan = ExecutionPlan(
            goal     = goal,
            tasks    = ordered,
            timeline = timeline,
            summary  = summary,
        )

        self._log("=== Plan Complete ===")
        return {
            "plan":          plan.to_dict(),
            "reasoning_log": self.reasoning_log,
            "condition":     condition,
            "description":   data["description"],
        }


# ─────────────────────────────────────────────
#  QUICK CLI TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    agent = PlannerAgent()
    result = agent.create_plan("Treatment plan for diabetes management")
    print(json.dumps(result, indent=2))
