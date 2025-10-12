from flask import Blueprint, request, jsonify, session
import google.generativeai as genai
import os

from flask_login import current_user

from models.subject import SUBJECTS

ai_bp = Blueprint("ai", __name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment.")
genai.configure(api_key=GEMINI_API_KEY)

conversations = {}


def build_gemini_model(system_prompt: str):
    is_guest = session.get("guest") and not getattr(current_user, "is_authenticated", False)
    max_tokens = 100 if is_guest else 500
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash-lite",
        system_instruction=system_prompt,
        generation_config={"temperature": 0.7, "max_output_tokens": max_tokens},
    )


def to_gemini_history(history):
    converted = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if not content:
            continue
        converted.append({
            "role": "model" if role == "assistant" else "user",
            "parts": [content],
        })
    return converted


@ai_bp.route("/api/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "")
    subject = data.get("subject", "math")
    session_id = data.get("session_id", "default")

    if not question.strip():
        return jsonify({"error": "Question cannot be empty"}), 400
    if subject not in SUBJECTS:
        subject = "math"

    history = conversations.setdefault(session_id, [])[-8:]

    try:
        system_prompt = (
            f"You are a helpful study assistant specializing in {SUBJECTS[subject]}. "
            f"Answer questions clearly and educationally. Keep responses concise but informative. "
            f"Focus only on {SUBJECTS[subject]} topics."
        )
        model = build_gemini_model(system_prompt)
        chat = model.start_chat(history=to_gemini_history(history))
        gemini_response = chat.send_message(question)
        answer = (gemini_response.text or "").strip()

        conversations[session_id].append({"role": "user", "content": question})
        conversations[session_id].append({"role": "assistant", "content": answer})
        conversations[session_id] = conversations[session_id][-8:]

        return jsonify({"answer": answer, "question": question, "subject": subject})
    except Exception as e:
        return jsonify({"error": f"Error generating response: {str(e)}"}), 500


@ai_bp.post("/api/verify")
def api_verify():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "default")
    subject = data.get("subject", "math")
    if subject not in SUBJECTS:
        subject = "math"
    history = conversations.get(session_id, [])
    if not history:
        return jsonify({"reply": "No previous Q&A available to verify."}), 200

    last_user_q = next(
        (m["content"] for m in reversed(history) if m.get("role") == "user"), None
    )
    if not last_user_q:
        return jsonify({"reply": "Couldn't find the last user question to verify."}), 200

    try:
        system_prompt = (
            f"You are a study assistant for {SUBJECTS[subject]}. "
            f"When asked to verify, respond with 3–6 reputable sources that support an answer "
            f"to the given question. Prefer URLs. If you are uncertain, say so."
        )
        model = build_gemini_model(system_prompt)
        prompt = (
            "Provide sources that support a correct answer to this question. "
            "Return a short bulleted list of sources with titles and URLs.\n\n"
            f"Question: {last_user_q}"
        )
        resp = model.generate_content(prompt)
        sources_text = (getattr(resp, "text", None) or "").strip() or "No sources found."
        return jsonify({"reply": sources_text}), 200
    except Exception as e:
        return jsonify({"reply": f"Error while verifying: {e}"}), 500


@ai_bp.post("/reset-conversation")
def reset_conversation():
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "default").strip()
    conversations.pop(session_id, None)
    return jsonify({"ok": True, "cleared_session_id": session_id})
