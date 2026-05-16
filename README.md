# MedAssist Chatbot

Web-based medical information chatbot powered by Gemini.

## Features
- Chat UI with send button and Enter-to-send (`Shift+Enter` for newline)
- Attachment upload from `+` menu:
  - `Camera` (device/browser dependent)
  - `Gallery` (images)
  - `Document` (PDF/TXT/CSV/JSON/MD)
- Text-only model behavior:
  - uploads are accepted and logged
  - model does not read binary/image/document content directly
  - user should type key details from uploaded content
- Theme toggle (`Light` / `Dark`)
- AutoCorrect toggle
- Draft save toggle (uses browser local storage)
- Emergency keyword guardrails

## Requirements
- Python 3.10+
- Gemini API key in `.env` (`GEMINI_API_KEY`)

## Install
```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Configure
Create or edit `.env`:
```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-1.5-flash
MOCK_LLM=False
PORT=5000
```

## Run Web App
```powershell
.\.venv\Scripts\activate
$env:PYTHONUTF8='1'
python app.py
```

Open: `http://127.0.0.1:5000`

## Run CLI (optional)
```powershell
.\.venv\Scripts\activate
python medicalaibot.py
```

## Notes
- The app only accesses files you explicitly select.
- Uploads are for interface/workflow only in text mode; the model cannot inspect attachment bytes.
- If UI looks stale, do a hard refresh (`Ctrl+F5`).
