# Corvanta Helpdesk Assistant

A FastAPI chat app for the supplied HR and IT policies. It cites its sources, uses an employee's country and grade when an ID is provided, asks when country is missing, and creates tickets only on request.

## Run locally

Requires Python 3.11 and a Gemini API key.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
pip install -r requirements.txt
```

Add your key to `.env`, then index the documents and start the app:

```powershell
python -m app.ingestion
uvicorn app.main:app --reload
```

Open the chat at [http://127.0.0.1:8000/](http://127.0.0.1:8000/). API docs are at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

The app uses Gemini for chat and embeddings, with a local Chroma index. Re-running ingestion updates the index. The 2026 leave policy replaces the 2025 version. Tickets are created only when requested; unsupported questions receive: `I couldn't find this in the provided documents.`

## API

`POST /chat` accepts `message`, `session_id`, and an optional `employee_id`. Reuse the session ID to continue a conversation.

## Tests

Run `pytest` from the project folder.

AI tools assisted with code and documentation.
