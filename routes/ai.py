import logging

from flask import Blueprint, request, jsonify, Response
import os, io, datetime
import google.generativeai as genai
import json

ai_bp = Blueprint("ai", __name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment.")
genai.configure(api_key=GEMINI_API_KEY)

conversations = {}
logger = logging.getLogger(__name__)

SUBJECTS = {
    'math': 'Mathematics',
    'biology': 'Biology',
    'chemistry': 'Chemistry',
    'physics': 'Physics',
    'history': 'History',
    'english': 'English Language',
    'spanish': 'Spanish Language',
    'french': 'French Language',
    'german': 'German Language',
    'chinese': 'Chinese Language'
}


def build_gemini_model(system_prompt: str):
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash-lite",
        system_instruction=system_prompt,
        generation_config={"temperature": 0.7, "max_output_tokens": 500},
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


@ai_bp.post("/api/ask")
def ask_question():
    data = request.get_json() or {}
    question = data.get('question', '')
    subject = data.get('subject', 'math')
    session_id = data.get('session_id', 'default')

    if not question.strip():
        return jsonify({'error': 'Question cannot be empty'}), 400
    if subject not in SUBJECTS:
        subject = 'math'

    history = conversations.setdefault(session_id, [])
    history_slice = history[-8:] if len(history) > 8 else history

    try:
        system_prompt = (
            f"You are a helpful study assistant specializing in {SUBJECTS[subject]}. "
            f"Answer questions clearly and educationally. Keep responses concise but informative. "
            f"Focus only on {SUBJECTS[subject]} topics."
        )
        model = build_gemini_model(system_prompt)
        chat = model.start_chat(history=to_gemini_history(history_slice))
        gemini_response = chat.send_message(question)
        answer = (gemini_response.text or "").strip()

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        if len(history) > 8:
            conversations[session_id] = history[-8:]

        return jsonify({'answer': answer, 'question': question, 'subject': subject})
    except Exception as e:
        logging.exception("Unable to ask question")
        return jsonify({'error': f'Error generating response: {str(e)}'}), 500


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

    last_user_q = next((m["content"] for m in reversed(history) if m.get("role") == "user"), None)
    if not last_user_q:
        return jsonify({"reply": "Couldn't find the last user question to verify."}), 200

    try:
        system_prompt = (
            f"You are a study assistant for {SUBJECTS[subject]}. "
            f"When asked to verify, respond with 3–6 reputable sources (books, papers, .edu/.gov/.org, or high quality sites) "
            f"that support an answer to the given question. Prefer URLs. If you are uncertain, say so."
        )
        model = build_gemini_model(system_prompt)
        prompt = (
            "Provide sources that support a clear, correct answer to this question. "
            "Return a short bulleted list of sources with titles and URLs.\n\n"
            f"Question: {last_user_q}"
        )
        resp = model.generate_content(prompt)
        sources_text = (getattr(resp, "text", None) or "").strip() or "No sources found."
        return jsonify({"reply": sources_text}), 200
    except Exception as e:
        logging.exception("Unable to verify question")
        return jsonify({"reply": f"Error while verifying: {e}"}), 500


@ai_bp.post("/reset-conversation")
def reset_conversation():
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "default").strip()
    conversations.pop(session_id, None)
    return jsonify({"ok": True, "cleared_session_id": session_id})


def recent_history_md(session_id: str):
    """Return last ~8 messages as Markdown."""
    hist = conversations.get(session_id, [])
    lines = []
    for m in hist:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content: continue
        prefix = "**You:**" if role == "user" else "**Assistant:**"
        lines.append(f"{prefix} {content}")
    return "\n\n".join(lines)


@ai_bp.post("/api/summary")
def summary_chat():
    data = request.get_json() or {}
    session_id = data.get("session_id", "default")
    subject = data.get("subject", "math")
    history = conversations.get(session_id, [])
    text_context = "\n\n".join([f"{m['role']}: {m['content']}" for m in history if m.get('content')])

    try:
        system_prompt = (
            f"You are a study assistant for {SUBJECTS.get(subject, subject)}. "
            "Summarize the conversation into 5-9 bullet points, highlighting steps, formulas, and definitions. "
            "Then list 8-15 key terms with brief one-line definitions. Use concise Markdown."
        )
        model = build_gemini_model(system_prompt)
        resp = model.generate_content(
            f"Conversation:\n{text_context}\n\nNow produce:\n- Summary bullets\n- Key terms (term — 1 line)")
        out = (getattr(resp, "text", None) or "").strip()
        # Split best-effort into two parts
        parts = out.split("Key", 1)
        summary = out if len(parts) == 1 else parts[0].strip()
        key_terms = "" if len(parts) == 1 else "Key" + parts[1]
        return jsonify({"summary": summary, "key_terms": key_terms})
    except Exception as e:
        logging.exception("Unable to get summary")
        return jsonify({"error": str(e)}), 500


@ai_bp.post("/api/quiz")
def quiz_chat():
    data = request.get_json() or {}
    session_id = data.get("session_id", "default")
    subject = data.get("subject", "math")
    n = int(data.get("num_questions", 5))
    history = conversations.get(session_id, [])
    text_context = "\n\n".join([f"{m['role']}: {m['content']}" for m in history if m.get('content')])

    try:
        system_prompt = (
            f"You are a quiz generator for {SUBJECTS.get(subject, subject)}. "
            f"Create {n} multiple-choice questions from the conversation (recent topics only). "
            "Format in Markdown as:\n\n"
            "### Q1\n"
            "A) ...\nB) ...\nC) ...\nD) ...\n\n**Answer:** C — short rationale\n\n"
            "Keep questions clear and leveled from easy to challenging."
        )
        model = build_gemini_model(system_prompt)
        resp = model.generate_content(f"Conversation:\n{text_context}\n\nGenerate the quiz now.")
        quiz_md = (getattr(resp, "text", None) or "").strip()
        return jsonify({"quiz_markdown": quiz_md})
    except Exception as e:
        logging.exception("Unable to post quiz")
        return jsonify({"error": str(e)}), 500


@ai_bp.get("/api/export")
def export_chat():
    session_id = request.args.get("session_id", "default")
    subject = request.args.get("subject", "math")
    history_md = recent_history_md(session_id)

    # quick inline summary to include
    try:
        system_prompt = (
            f"You are a study assistant for {SUBJECTS.get(subject, subject)}. "
            "Write a short executive summary (6–10 bullets) of the conversation. Use Markdown."
        )
        model = build_gemini_model(system_prompt)
        resp = model.generate_content(f"Conversation:\n{history_md}")
        summary_md = (getattr(resp, "text", None) or "").strip()
    except Exception:
        logging.exception("Unable to export")
        summary_md = "_Summary unavailable._"

    now = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H%M%SZ")
    filename = f"chat_export_{subject}_{now}.md"
    md = f"# Study Chat Export — {subject}\n\n## Summary\n\n{summary_md}\n\n---\n\n## Conversation\n\n{history_md}\n"
    buf = io.BytesIO(md.encode("utf-8"))
    return Response(
        buf.getvalue(),
        headers={
            "Content-Type": "text/markdown; charset=utf-8",
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@ai_bp.post("/api/quiz_json")
def quiz_json():
    import re, json, logging

    data = request.get_json() or {}
    session_id = data.get("session_id", "default")
    subject = data.get("subject", "math")
    n = int(data.get("num_questions", 5))

    history = conversations.get(session_id, [])
    ctx = "\n\n".join(
        f"{m['role']}: {m['content']}"
        for m in history[-10:] if m.get('content')
    )

    system_prompt = (
        f"You are a quiz generator for {SUBJECTS.get(subject, subject)}. "
        f"Write {n} multiple-choice questions based on the conversation. "
        "Each question must have exactly 4 choices. Include a 0-based 'answer_index' "
        "and a one-sentence 'rationale' explaining the correct answer. "
        "Return ONLY valid JSON with this schema:\n"
        '{ "questions": [ { "question": "string", "choices": ["string","string","string","string"], "answer_index": 0, "rationale": "string" } ] }'
    )

    try:
        model = build_gemini_model(system_prompt)
        resp = model.generate_content(
            f"Conversation:\n{ctx}\n\nNow produce ONLY valid JSON (no markdown, no ```json)."
        )
        raw = (getattr(resp, "text", None) or "").strip()

        cleaned = raw
        cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip(), flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as je:
            logging.warning("Failed JSON parse: %s", je)
            return jsonify({
                "error": "Gemini returned invalid JSON. Try again.",
                "raw": raw[:1000]  # send a snippet for debugging
            }), 500

        qs = payload.get("questions", [])
        if not isinstance(qs, list) or len(qs) == 0:
            return jsonify({"error": "No questions returned", "raw": raw}), 500

        payload["questions"] = qs[:n]
        return jsonify(payload)

    except Exception as e:
        logging.exception("Unable to generate quiz JSON")
        msg = str(e)
        #if "429" in msg or "quota" in msg.lower():
        #    return jsonify({
        #        "error": "Gemini API quota exceeded — please wait or enable billing.",
        #        "detail": msg
        #    }), 429
        dummy_payload = {
            "questions": [
                {
                    "question": "Which planet is known as the Red Planet?",
                    "choices": ["Earth", "Mars", "Venus", "Jupiter"],
                    "answer_index": 1,
                    "rationale": "Mars appears red because of iron oxide (rust) on its surface."
                },
                {
                    "question": "What is the chemical symbol for water?",
                    "choices": ["H2O", "O2", "HO", "H3O"],
                    "answer_index": 0,
                    "rationale": "Water consists of two hydrogen atoms and one oxygen atom."
                },
                {
                    "question": "Who wrote the play 'Romeo and Juliet'?",
                    "choices": ["William Shakespeare", "Charles Dickens", "Leo Tolstoy", "Mark Twain"],
                    "answer_index": 0,
                    "rationale": "‘Romeo and Juliet’ was written by William Shakespeare in the late 16th century."
                },
                {
                    "question": "Which gas do plants absorb during photosynthesis?",
                    "choices": ["Oxygen", "Carbon Dioxide", "Nitrogen", "Hydrogen"],
                    "answer_index": 1,
                    "rationale": "Plants use carbon dioxide (CO₂) and sunlight to produce glucose and oxygen."
                },
                {
                    "question": "What is the largest ocean on Earth?",
                    "choices": ["Atlantic Ocean", "Indian Ocean", "Pacific Ocean", "Arctic Ocean"],
                    "answer_index": 2,
                    "rationale": "The Pacific Ocean covers about one-third of Earth’s surface, making it the largest."
                }
            ]
        }
        return jsonify(dummy_payload)

        #return jsonify({"error": f"quiz generation failed: {msg}"}), 500

