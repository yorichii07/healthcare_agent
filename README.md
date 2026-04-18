# Healthcare Planning Assistant Agent

A full-stack healthcare planning app that converts a user goal into a structured execution plan.

- Backend: FastAPI + Python agent orchestration
- AI runtime: Groq via LangChain (with graceful fallback mode)
- Frontend: HTML/CSS/JS single-page interface

## Features

- Goal-based plan generation (`/api/plan`)
- Three planning modes:
  - `auto`: AI first, fallback logic when unavailable
  - `ai`: strict LLM planning only
  - `fallback`: deterministic rules from healthcare knowledge base
- Task-level timeline, dependencies, priority, and validation status
- Mock tool integrations for:
  - doctor availability
  - medicine stock checks
  - lab test availability
- Backend health status shown in UI

## Project Structure

```text
health_agent/
  backend/
    planner_agent.py
    requirements.txt
    server.py
  frontend/
    index.html
    css/
      style.css
    js/
      script.js
  .env.example
  .gitignore
```

## Prerequisites

- Python 3.10+
- A Groq API key (only required for `auto`/`ai` AI planning)

## Setup

1. Clone the repository

```bash
git clone https://github.com/yorichii07/healthcare_agent.git
cd healthcare_agent
```

2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install backend dependencies

```bash
pip install -r backend/requirements.txt
```

4. Configure environment variables

```bash
cp .env.example .env
```

Set values in `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

## Run

Start the backend server from the project root:

```bash
python -m uvicorn backend.server:app --reload
```

Open the app in your browser:

- http://127.0.0.1:8000/

The FastAPI backend serves the frontend and exposes APIs under `/api`.

## API Endpoints

### `GET /api/`
Health message endpoint.

Example response:

```json
{
  "message": "Healthcare Planning Assistant Agent API is running"
}
```

### `POST /api/plan`
Generates a healthcare execution plan.

Request body:

```json
{
  "goal": "Treatment plan for diabetes",
  "mode": "auto"
}
```

- `goal` (string, required): user intent
- `mode` (string, optional): `auto` | `ai` | `fallback` (default `auto`)

### `GET /api/conditions`
Returns available knowledge-base conditions and task counts.

## Troubleshooting

### Backend not reachable in UI
- Ensure server is running at `http://127.0.0.1:8000`
- Open the app via backend root URL, not by directly opening `frontend/index.html`

### AI mode unavailable (`503`)
- Verify `.env` has a valid `GROQ_API_KEY`
- Restart server after changing environment variables
- Confirm dependencies installed in active virtual environment

### Model deprecated / provider unavailable
- Update `GROQ_MODEL` in `.env` to a supported model
- Retry later or use `auto`/`fallback`

## Security Notes

- Never commit `.env` files or API keys
- Rotate any API key immediately if exposed
- Keep `.env.example` as placeholder values only

## Disclaimer

This project is for educational/demo purposes and does not replace professional medical diagnosis or treatment.