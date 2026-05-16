import os
import json
import time
import datetime

GEMINI_IMPORT_ERROR = None
try:
    import google.generativeai as genai
except Exception as exc:
    genai = None
    GEMINI_IMPORT_ERROR = exc

from dotenv import load_dotenv


load_dotenv(override=False)


def _normalize_api_key(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().strip('"').strip("'").strip()


def _resolve_gemini_api_key() -> str:
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


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


GEMINI_API_KEY = _resolve_gemini_api_key()
MODEL = (os.environ.get("GEMINI_MODEL") or "gemini-1.5-flash").strip()
TEMPERATURE = 0.4
LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 4, minimum=1)

SYSTEM_PROMPT = """
You are MedAssist AI, a medical information assistant for real-time text chat.

PRIMARY GOAL:
- Give clear, safe, practical help in a friendly conversational style.

VOICE AND TONE:
- Natural, calm, respectful, and human.
- Concise-first: answer quickly, then add only needed detail.
- Plain language; explain medical terms briefly when used.

SAFETY RULES:
- You are not a doctor; do not diagnose or prescribe.
- Use cautious language: "possible", "may help", "often used".
- If emergency red flags appear (chest pain, stroke signs, severe breathing trouble,
  heavy bleeding, suicidal thoughts), clearly instruct immediate emergency care.
- For medication dosage questions, advise confirming with a licensed clinician or pharmacist.
""".strip()

EMERGENCY_KEYWORDS = [
    "chest pain",
    "heart attack",
    "can't breathe",
    "cannot breathe",
    "difficulty breathing",
    "stroke",
    "unconscious",
    "unresponsive",
    "severe bleeding",
    "overdose",
    "suicide",
    "kill myself",
    "not breathing",
    "seizure",
    "anaphylaxis",
]


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
            gemini_msgs.append({"role": role, "parts": [msg["content"]]})
        return gemini_msgs

    def latest_user_message(self) -> str:
        for msg in reversed(self.history):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    def save(self, filename: str = "medassist_session.json") -> str:
        data = {
            "app": "MedAssist AI CLI",
            "model": MODEL,
            "saved_at": datetime.datetime.now().isoformat(),
            "messages": self.history,
        }
        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        return filename


class MedicalLLMClient:
    def __init__(self, api_key: str):
        if genai is None:
            raise RuntimeError(_gemini_dependency_hint(GEMINI_IMPORT_ERROR))
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=MODEL,
            generation_config={"temperature": TEMPERATURE},
            system_instruction=SYSTEM_PROMPT,
        )

    def _raise_error(self, raw_error: Exception, retries: int):
        err = str(raw_error).lower()
        if "quota" in err or "rate" in err:
            raise RuntimeError("Rate limit reached. Please wait and try again.")
        if "api key" in err or "authentication" in err or "invalid" in err:
            raise RuntimeError("Invalid API key. Check GEMINI_API_KEY.")
        if "not found" in err and "model" in err:
            raise RuntimeError(
                f"Model '{MODEL}' not found. Set GEMINI_MODEL to a valid text model, "
                "for example gemini-1.5-flash."
            )
        raise RuntimeError(f"API error after {retries} attempts: {raw_error}")

    def chat(self, conversation: ConversationManager, retries: int = LLM_MAX_RETRIES) -> str:
        history = conversation.to_gemini_history()
        user_text = conversation.latest_user_message()

        for attempt in range(1, retries + 1):
            try:
                chat = self.model.start_chat(history=history)
                response = chat.send_message(user_text)
                text = (getattr(response, "text", "") or "").strip()
                if text:
                    return text
                return "I could not generate a response. Please rephrase your question."
            except Exception as exc:
                if attempt < retries:
                    time.sleep(min(8, 1.5 * attempt))
                    continue
                self._raise_error(exc, retries)

        raise RuntimeError("Failed after retries.")


class SafetyLayer:
    @staticmethod
    def is_emergency(text: str) -> bool:
        lower = text.lower()
        return any(keyword in lower for keyword in EMERGENCY_KEYWORDS)


def main():
    print("MedAssist AI CLI (text chat). Type 'exit' to quit.")

    if genai is None:
        print(f"ERROR: {_gemini_dependency_hint(GEMINI_IMPORT_ERROR)}")
        return
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not found in .env or environment.")
        return

    conversation = ConversationManager()
    llm = MedicalLLMClient(api_key=GEMINI_API_KEY)
    safety = SafetyLayer()

    welcome = (
        "Hello. I am MedAssist AI.\n"
        "I provide general medical guidance from text only.\n"
        "I am not a doctor. For personal decisions, consult a licensed clinician."
    )
    print(welcome)
    conversation.add("assistant", welcome)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ("exit", "quit"):
                break
            if not user_input:
                continue

            if safety.is_emergency(user_input):
                print(
                    "\nEMERGENCY DETECTED: Call 911 now or go to the nearest emergency room immediately."
                )
                continue

            conversation.add("user", user_input)
            try:
                reply = llm.chat(conversation)
            except RuntimeError as exc:
                reply = f"Error: {exc}"

            conversation.add("assistant", reply)
            print(f"\nMedAssist AI: {reply}")
        except KeyboardInterrupt:
            print("\nSession ended by user.")
            break
        except Exception as exc:
            print(f"\nUnexpected error: {exc}")

    print("\nSession ended.")


if __name__ == "__main__":
    main()
