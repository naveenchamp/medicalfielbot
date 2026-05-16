import os
import time
import datetime
import mimetypes
import secrets
import random
import re
import threading
GEMINI_IMPORT_ERROR = None
try:
    import google.generativeai as genai
except Exception as e:
    genai = None
    GEMINI_IMPORT_ERROR = e
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
#  CONFIG  (NOW USING ENVIRONMENT VARIABLES)
# ─────────────────────────────────────────────────────────────
load_dotenv(override=False)

def _normalize_api_key(value: str | None) -> str:
    if not value:
        return ""
    # Remove accidental spaces/quotes from .env value
    return value.strip().strip('"').strip("'").strip()

def _resolve_gemini_api_key() -> str:
    # Support common env names used in Google examples
    candidates = [
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("GOOGLE_API_KEY"),
        os.environ.get("API_KEY"),
    ]
    for raw in candidates:
        key = _normalize_api_key(raw)
        if key:
            return key
    return ""

def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default

def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        return default

def _gemini_dependency_hint(err: Exception | None) -> str:
    if err is None:
        return "Gemini client library not installed."
    message = str(err)
    if "filename or extension is too long" in message.lower():
        return (
            "Gemini dependency import failed due to Windows path-length DLL loading. "
            "Move project to a short path (for example C:\\dev\\medicalfieldbot), "
            "recreate .venv, and reinstall requirements."
        )
    return f"Gemini dependency import failed: {message}"

GEMINI_API_KEY = _resolve_gemini_api_key()

# Use a configurable text model with a compatibility-safe default.
MODEL = (os.environ.get("GEMINI_MODEL") or "gemini-1.5-flash").strip()
TEMPERATURE = 0.4

# Local testing flag: when true the server returns mock replies instead of calling the LLM
MOCK_LLM = os.environ.get("MOCK_LLM", "false").lower() in ("1", "true", "yes")
LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 6, minimum=1)
LLM_BASE_BACKOFF_SECONDS = _env_float("LLM_BASE_BACKOFF_SECONDS", 1.0, minimum=0.1)
LLM_MAX_BACKOFF_SECONDS = _env_float("LLM_MAX_BACKOFF_SECONDS", 16.0, minimum=1.0)
LLM_CONCURRENCY = _env_int("LLM_CONCURRENCY", 2, minimum=1)
SESSION_MIN_INTERVAL_SECONDS = _env_float("SESSION_MIN_INTERVAL_SECONDS", 0.9, minimum=0.0)
SESSION_ACTIVITY_TTL_SECONDS = _env_float("SESSION_ACTIVITY_TTL_SECONDS", 1200.0, minimum=60.0)

SYSTEM_PROMPT = """
You are MedAssist AI, a medical information assistant for real-time chat.

PRIMARY GOAL:
- Give clear, safe, practical help in a friendly conversational style.

VOICE AND TONE:
- Natural, calm, respectful, and human.
- Concise-first: answer quickly, then add only needed detail.
- Plain language; explain medical terms briefly when used.
- No filler openings (avoid "Great question", "I'd be happy to help").

SAFETY RULES:
- You are not a doctor; do not diagnose or prescribe.
- Use cautious language: "possible", "may help", "often used".
- If emergency red flags appear (chest pain, stroke signs, severe breathing trouble,
  heavy bleeding, suicidal thoughts), clearly instruct immediate emergency care.
- For medication dosage questions, advise confirming with a licensed clinician or pharmacist.

DEFAULT RESPONSE SHAPE (unless user asks otherwise):
1) One direct answer sentence.
2) 3-5 short, actionable bullets.
3) One short follow-up question only if it helps personalize next steps.

LENGTH TARGET:
- Typical response: 60-120 words.
- Complex/risky topics: up to 180 words.

QUALITY BAR:
- Prioritize what the user should do now.
- Include practical at-home steps when safe.
- State when to seek care, with concrete warning signs.
- If uncertain, say so briefly and give the safest next action.
""".strip()

EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "can't breathe", "cannot breathe",
    "difficulty breathing", "stroke", "unconscious", "unresponsive",
    "severe bleeding", "overdose", "suicide", "kill myself",
    "not breathing", "seizure", "anaphylaxis",
]

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
SUPPORTED_IMAGE_MIME_PREFIX = "image/"
SUPPORTED_DOCUMENT_MIME_TYPES = {
    "application/pdf",
    "text/plain",
    "text/csv",
    "application/json",
    "text/markdown",
}
SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".json": "application/json",
    ".md": "text/markdown",
}
TEXT_ONLY_UPLOAD_NOTE = (
    "Upload received. This assistant runs in text-only mode and cannot read file/photo "
    "content directly. Please type the key details you want reviewed."
)

print("Configuration loaded.")

# Configure Gemini once if available
if genai is None:
    print(f"Warning: {_gemini_dependency_hint(GEMINI_IMPORT_ERROR)}")
else:
    if not GEMINI_API_KEY:
        print("Warning: GEMINI API key not found. The API will run with model disabled.")
    else:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
        except Exception as e:
            print(f"Warning: Failed to configure Gemini client: {e}")

# ─────────────────────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='static', static_url_path='')
app.secret_key = secrets.token_hex(16)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + (256 * 1024)
CORS(app)

# Store chat sessions
chat_sessions = {}
session_last_request_at = {}
session_request_lock = threading.Lock()
llm_call_gate = threading.BoundedSemaphore(LLM_CONCURRENCY)


def _now_iso() -> str:
    return datetime.datetime.now().isoformat()


def _json_error(response_text: str, status_code: int = 500, detail: str | None = None):
    payload = {
        "response": response_text,
        "is_emergency": False,
        "error": True,
        "timestamp": _now_iso(),
    }
    if detail:
        payload["detail"] = detail
    return jsonify(payload), status_code


def _emergency_response():
    return jsonify({
        "response": (
            "EMERGENCY DETECTED.\n\n"
            "Please call 911 immediately or go to your nearest emergency room. Do not wait.\n\n"
            "This is an automated emergency detection message."
        ),
        "is_emergency": True,
        "timestamp": _now_iso(),
    })


def _get_conversation(session_id: str) -> "ConversationManager":
    session_key = (session_id or "").strip() or "default"
    if session_key not in chat_sessions:
        chat_sessions[session_key] = ConversationManager()
    return chat_sessions[session_key]


def _pace_session_request(session_id: str):
    session_key = (session_id or "").strip() or "default"
    if SESSION_MIN_INTERVAL_SECONDS <= 0:
        return

    now = time.time()
    wait_seconds = 0.0
    with session_request_lock:
        previous = session_last_request_at.get(session_key)
        if previous is not None:
            elapsed = now - previous
            if elapsed < SESSION_MIN_INTERVAL_SECONDS:
                wait_seconds = SESSION_MIN_INTERVAL_SECONDS - elapsed

        session_last_request_at[session_key] = now + wait_seconds

        if len(session_last_request_at) > 1000:
            stale_before = now - SESSION_ACTIVITY_TTL_SECONDS
            for key, ts in list(session_last_request_at.items()):
                if ts < stale_before:
                    session_last_request_at.pop(key, None)

    if wait_seconds > 0:
        time.sleep(wait_seconds)


def _to_runtime_error_response(err: RuntimeError):
    msg = str(err)
    if "invalid api key" in msg.lower() or "authentication" in msg.lower():
        return _json_error(
            "Invalid or missing GEMINI_API_KEY. Check your .env and restart.",
            status_code=401,
            detail=msg,
        )
    return _json_error(f"Runtime error: {msg}", status_code=502)


# ─────────────────────────────────────────────────────────────
#  CONVERSATION MANAGER CLASS
# ─────────────────────────────────────────────────────────────
class ConversationManager:
    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.history: list[dict] = []

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def to_gemini_history(self) -> list[dict]:
        gemini_msgs = []
        for msg in self.history[:-1]:
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_msgs.append({
                "role": role,
                "parts": [msg["content"]]
            })
        return gemini_msgs

    def latest_user_message(self) -> str:
        for msg in reversed(self.history):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    def clear(self):
        self.history.clear()


# ─────────────────────────────────────────────────────────────
#  LLM CLIENT CLASS
# ─────────────────────────────────────────────────────────────
class MedicalLLMClient:
    def __init__(self):
        self.enabled = False
        self.disabled_reason = "LLM client not initialized."

        if genai is None:
            self.disabled_reason = _gemini_dependency_hint(GEMINI_IMPORT_ERROR)
            return
        if not GEMINI_API_KEY:
            self.disabled_reason = (
                "GEMINI_API_KEY not found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env "
                "and restart the server."
            )
            return

        try:
            if hasattr(genai, "GenerativeModel"):
                self.model = genai.GenerativeModel(
                    model_name=MODEL,
                    generation_config={
                        "temperature": TEMPERATURE,
                    },
                    system_instruction=SYSTEM_PROMPT
                )
                self.enabled = True
                self.disabled_reason = None
            else:
                self.disabled_reason = "Installed Gemini client does not expose GenerativeModel; update the package."
        except Exception as e:
            self.disabled_reason = f"Failed to initialize LLM client: {e}"

    @staticmethod
    def _raise_runtime_error(raw_error: Exception, retries: int):
        err_str = str(raw_error).lower()
        if "quota" in err_str or "rate" in err_str:
            raise RuntimeError("Rate limit hit. Please wait and try again.")
        if "api key" in err_str or "authentication" in err_str or "invalid" in err_str:
            raise RuntimeError("Invalid API key. Please check your GEMINI_API_KEY environment variable.")
        if "not found" in err_str and "model" in err_str:
            raise RuntimeError(
                f"Model '{MODEL}' was not found. Set GEMINI_MODEL to a supported text model "
                "(example: gemini-1.5-flash)."
            )
        raise RuntimeError(f"API error after {retries} attempts: {raw_error}")

    @staticmethod
    def _is_non_retryable_error(raw_error: Exception) -> bool:
        err = str(raw_error).lower()
        return any(token in err for token in (
            "api key", "authentication", "invalid",
            "not found",
        ))

    @staticmethod
    def _extract_retry_after(raw_error: Exception) -> float | None:
        err = str(raw_error).lower()
        match = re.search(r"retry(?:[- ]after| in)?[^0-9]{0,12}(\d+)", err)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    @classmethod
    def _backoff_seconds(cls, raw_error: Exception, attempt: int) -> float:
        retry_after = cls._extract_retry_after(raw_error)
        if retry_after is not None:
            return min(LLM_MAX_BACKOFF_SECONDS, max(0.5, retry_after + random.uniform(0, 0.4)))

        base = LLM_BASE_BACKOFF_SECONDS * (2 ** max(0, attempt - 1))
        jitter = random.uniform(0, max(0.1, LLM_BASE_BACKOFF_SECONDS))
        return min(LLM_MAX_BACKOFF_SECONDS, base + jitter)

    def chat(self, conversation: ConversationManager, retries: int = LLM_MAX_RETRIES) -> str:
        history = conversation.to_gemini_history()
        user_text = conversation.latest_user_message()
        if not self.enabled:
            reason = self.disabled_reason
            return (
                "Model unavailable. "
                f"{reason} \n\n"
                "The API is running but the language model is disabled. "
                "Please check /health for detailed diagnostics."
            )

        for attempt in range(1, retries + 1):
            try:
                with llm_call_gate:
                    chat = self.model.start_chat(history=history)
                    response = chat.send_message(user_text)
                reply_text = (getattr(response, "text", "") or "").strip()
                if reply_text:
                    return reply_text
                return "I could not generate a response. Please rephrase your message."
            except Exception as e:
                if self._is_non_retryable_error(e):
                    self._raise_runtime_error(e, retries)
                if attempt < retries:
                    time.sleep(self._backoff_seconds(e, attempt))
                    continue
                self._raise_runtime_error(e, retries)
        raise RuntimeError("Failed after retries.")

# ─────────────────────────────────────────────────────────────
#  SAFETY LAYER
# ─────────────────────────────────────────────────────────────
class SafetyLayer:
    @staticmethod
    def is_emergency(text: str) -> bool:
        lower = text.lower()
        return any(kw in lower for kw in EMERGENCY_KEYWORDS)


# Initialize components
llm = MedicalLLMClient()
safety = SafetyLayer()


# ─────────────────────────────────────────────────────────────
#  FLASK ROUTES
# ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get('message') or '').strip()
    session_id = (data.get('session_id') or 'default').strip()

    if not user_message:
        return _json_error("Message is required.", status_code=400)

    if safety.is_emergency(user_message):
        return _emergency_response()

    conversation = _get_conversation(session_id)
    conversation.add("user", user_message)

    if MOCK_LLM:
        mock_reply = (
            "[MOCK] This is a mock response for testing purposes. "
            f"You said: {user_message}"
        )
        conversation.add("assistant", mock_reply)
        return jsonify({
            'response': mock_reply,
            'is_emergency': False,
            'mock': True,
            'timestamp': _now_iso()
        })

    try:
        _pace_session_request(session_id)
        reply = llm.chat(conversation)
        conversation.add("assistant", reply)
        return jsonify({
            'response': reply,
            'is_emergency': False,
            'timestamp': _now_iso()
        })
    except RuntimeError as e:
        return _to_runtime_error_response(e)
    except Exception as e:
        return _json_error(f"Unexpected error: {str(e)}", status_code=500)


@app.route('/api/reset', methods=['POST'])
def reset_chat():
    data = request.get_json(silent=True) or {}
    session_id = (data.get('session_id') or 'default').strip()
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    with session_request_lock:
        session_last_request_at.pop(session_id or "default", None)
    return jsonify({'status': 'success', 'timestamp': _now_iso()})


@app.route('/api/chat/upload', methods=['POST'])
def chat_with_upload():
    session_id = (request.form.get('session_id') or 'default').strip()
    user_message = (request.form.get('message') or '').strip()

    attachment, error = _read_attachment_from_request()
    if error:
        return _json_error(error, status_code=400)

    if user_message and safety.is_emergency(user_message):
        return _emergency_response()

    conversation = _get_conversation(session_id)

    user_log_message = user_message or "I uploaded a file or photo. Please guide me from text only."
    user_log_message = (
        f"{user_log_message}\n\n"
        f"[Attachment uploaded: {attachment['filename']} | {attachment['mime_type']} | {attachment['kind']}]\n"
        "[Important: This is a text-only assistant. You cannot read attachment bytes. "
        "Ask the user to type relevant details from the file/photo.]"
    )
    conversation.add("user", user_log_message)

    if MOCK_LLM:
        mock_reply = (
            "[MOCK] "
            f"{TEXT_ONLY_UPLOAD_NOTE}\n"
            f"Attachment name: {attachment['filename']} ({attachment['mime_type']})."
        )
        conversation.add("assistant", mock_reply)
        return jsonify({
            'response': mock_reply,
            'is_emergency': False,
            'mock': True,
            'text_model_only': True,
            'attachment': {
                'filename': attachment['filename'],
                'mime_type': attachment['mime_type'],
                'kind': attachment['kind'],
            },
            'timestamp': _now_iso()
        })

    try:
        _pace_session_request(session_id)
        reply = llm.chat(conversation)
        reply = f"{TEXT_ONLY_UPLOAD_NOTE}\n\n{reply}"
        conversation.add("assistant", reply)
        return jsonify({
            'response': reply,
            'is_emergency': False,
            'text_model_only': True,
            'attachment': {
                'filename': attachment['filename'],
                'mime_type': attachment['mime_type'],
                'kind': attachment['kind'],
            },
            'timestamp': _now_iso()
        })
    except RuntimeError as e:
        return _to_runtime_error_response(e)
    except Exception as e:
        return _json_error(f"Unexpected error: {str(e)}", status_code=500)


def _read_attachment_from_request():
    uploaded = request.files.get("file")
    if uploaded is None:
        return None, "No file uploaded."

    filename = (uploaded.filename or "").strip()
    if not filename:
        return None, "Missing filename."

    file_bytes = uploaded.read() or b""
    if not file_bytes:
        return None, "Uploaded file is empty."
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        return None, f"File is too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."

    def _detect_mime_type(filename: str, provided_mime: str | None) -> str:
        mime_type = (provided_mime or "").strip().lower()
        if mime_type:
            return mime_type
        guessed, _ = mimetypes.guess_type(filename or "")
        return (guessed or "application/octet-stream").lower()

    mime_type = _detect_mime_type(filename, uploaded.mimetype)

    def _attachment_kind(filename: str, mime_type: str) -> str | None:
        if mime_type.startswith(SUPPORTED_IMAGE_MIME_PREFIX):
            return "image"
        if mime_type in SUPPORTED_DOCUMENT_MIME_TYPES:
            return "document"
        ext = os.path.splitext(filename or "")[1].lower()
        if ext in SUPPORTED_DOCUMENT_EXTENSIONS:
            return "document"
        return None

    kind = _attachment_kind(filename, mime_type)
    if kind is None:
        return None, "Unsupported file type. Use image, PDF, TXT, CSV, JSON, or MD files."

    return {
        "filename": filename,
        "mime_type": mime_type,
        "size_bytes": len(file_bytes),
        "kind": kind,
    }, None


@app.route('/health')
def health():
    gemini_info = {
        'client_installed': genai is not None,
        'llm_enabled': getattr(llm, 'enabled', False),
        'disabled_reason': getattr(llm, 'disabled_reason', None),
        'import_error': str(GEMINI_IMPORT_ERROR) if GEMINI_IMPORT_ERROR else None,
        'model': MODEL,
        'text_model_only': True,
        'supports_upload_interface': True,
        'reads_attachment_content': False,
    }
    return jsonify({'status': 'healthy', 'gemini': gemini_info, 'mock_mode': MOCK_LLM})


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
