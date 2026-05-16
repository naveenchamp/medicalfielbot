# Deploy MedAssist Chat to Render

This project already includes a Flask web app in `app.py`.

## 1. Required Files

- `app.py`
- `requirements.txt`
- `static/index.html`
- `static/theme.js`

## 2. Render Service Settings

- Environment: `Python`
- Build Command: `pip install -r requirements.txt`
- Start Command: `python app.py`

## 3. Environment Variables

Set these in Render dashboard:

- `GEMINI_API_KEY`: your API key
- `GEMINI_MODEL`: `gemini-1.5-flash` (or another supported text model)
- `MOCK_LLM`: `false`
- `PORT`: Render sets this automatically, so optional

## 4. API Endpoints

- `POST /api/chat`
  - JSON body: `{"message":"...", "session_id":"optional"}`
- `POST /api/chat/upload`
  - form-data: `message`, `session_id`, `file`
  - Note: upload is accepted, but this app runs in text-only model mode and does not read attachment bytes.
- `GET /health`

## 5. Quick Test

```bash
curl -X POST https://<your-render-url>/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"What are common flu symptoms?\"}"
```
