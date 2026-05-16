import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
model_name = (os.getenv("GEMINI_MODEL") or "gemini-1.5-flash").strip()

if not api_key:
    print("No API key (set GEMINI_API_KEY or GOOGLE_API_KEY).")
else:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    try:
        models = genai.list_models()
        for model in models:
            print(model.name)
        model = genai.GenerativeModel(model_name)
        chat = model.start_chat(history=[])
        response = chat.send_message("Hello")
        print("Success:", response.text)
    except Exception as e:
        print("Error:", e)
