import logging
import os, io, datetime, re, json
from flask import Blueprint, request, jsonify, Response
import requests
import google.generativeai as genai

ai_bp = Blueprint("ai", __name__)

# ---------- Gemini setup ----------
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

# ---------- Gemini helpers ----------
def build_gemini_model(system_prompt: str):
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash-lite",
        system_instruction=system_prompt,
        generation_config={"temperature": 0.6, "max_output_tokens": 600},
    )

def build_quiz_model(system_prompt: str, response_mime_type: str, max_output_tokens: int = 1200, temperature: float = 0.7):
    cfg = {"temperature": temperature, "max_output_tokens": max_output_tokens}
    if response_mime_type:
        cfg["response_mime_type"] = response_mime_type
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash-lite",
        system_instruction=system_prompt,
        generation_config=cfg,
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

# ---------- Wolfram helpers (no verdicts) ----------
def _wolfram_primary_from_pods(j: dict) -> str:
    try:
        pods = j["queryresult"]["pods"]
    except Exception:
        return ""
    primary = next((p for p in pods if p.get("primary")), None)
    candidates = [primary] if primary else pods[:1]
    for p in candidates or []:
        for sp in p.get("subpods", []):
            pt = sp.get("plaintext")
            if pt:
                return pt.strip()
    for p in pods:
        for sp in p.get("subpods", []):
            pt = sp.get("plaintext")
            if pt:
                return pt.strip()
    return ""

def wolfram_answer(appid: str, query: str, timeout_result=8, timeout_query=12) -> str:
    """Ask Wolfram|Alpha: try /v2/result first, then /v2/query (JSON)."""
    if not appid or not query:
        return ""
    # 1) short answer
    try:
        r = requests.get(
            "https://api.wolframalpha.com/v2/result",
            params={"i": query, "appid": appid},
            timeout=timeout_result,
        )
        if r.status_code == 200 and r.text:
            return r.text.strip()
    except requests.RequestException:
        pass
    # 2) JSON fallback with reinterpret
    try:
        r = requests.get(
            "https://api.wolframalpha.com/v2/query",
            params={"input": query, "appid": appid, "output": "json", "reinterpret": "true"},
            timeout=timeout_query,
        )
        if r.ok:
            j = r.json()
            return _wolfram_primary_from_pods(j) or ""
    except requests.RequestException:
        pass
    return ""

# ---------- Routes ----------
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
    """
    Show Gemini sources FIRST, then Wolfram|Alpha answer — NO VERDICT.
    Sections (Markdown):
      - Question
      - Assistant Answer (excerpt)
      - Gemini Suggested Sources
      - Wolfram|Alpha Answer
    """
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "default")
    subject = data.get("subject", "math")
    if subject not in SUBJECTS:
        subject = "math"

    history = conversations.get(session_id, [])
    provided_question = (data.get("question") or "").strip()
    answer_excerpt = (data.get("answer_excerpt") or "").strip()
    last_user_q = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
    last_assistant = next((m["content"] for m in reversed(history) if m.get("role") == "assistant"), "")
    question = provided_question or last_user_q
    if not answer_excerpt:
        answer_excerpt = (last_assistant or "")[:400]

    if not question:
        return jsonify({"reply": "No question available to verify."}), 200

    # 1) Gemini sources
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
            f"Question: {question}"
        )
        resp = model.generate_content(prompt)
        sources_text = (getattr(resp, "text", None) or "").strip() or "No sources found."
    except Exception as e:
        logger.exception("Unable to fetch Gemini sources")
        sources_text = f"_Error while fetching sources: {e}_"

    # 2) Wolfram
    appid = os.environ.get("WOLFRAM_APPID")
    if not appid:
        wa_display = "_No Wolfram AppID configured — set WOLFRAM_APPID._"
    else:
        wa_text = wolfram_answer(appid, question) or ""
        wa_display = f"`{wa_text}`" if wa_text else "_No Wolfram|Alpha result for this query._"

    md = (
        f"### Question\n"
        f"`{question}`\n\n"
        f"### Assistant Answer (excerpt)\n"
        f"`{(answer_excerpt or '').strip()}`\n\n"
        f"### Gemini Suggested Sources\n"
        f"{sources_text}\n\n"
        f"### Wolfram|Alpha Answer\n"
        f"{wa_display}\n"
    )
    return jsonify({"reply": md}), 200

@ai_bp.post("/reset-conversation")
def reset_conversation():
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "default").strip()
    conversations.pop(session_id, None)
    return jsonify({"ok": True, "cleared_session_id": session_id})

def recent_history_md(session_id: str):
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

# ---------- JSON quiz endpoints (interactive UI consumes /api/quiz_json) ----------
@ai_bp.post("/api/quiz_json1")
def quiz_json1():
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
                "raw": raw[:1000]
            }), 500

        qs = payload.get("questions", [])
        if not isinstance(qs, list) or len(qs) == 0:
            return jsonify({"error": "No questions returned", "raw": raw}), 500

        payload["questions"] = qs[:n]
        return jsonify(payload)

    except Exception as e:
        logging.exception("Unable to generate quiz JSON")
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

@ai_bp.post("/api/quiz_json")
def quiz_json():
    import re, json, logging

    def dummy_quiz(n=5):
        return {
            "questions": [
                {
                    "question": "Which planet is known as the Red Planet?",
                    "choices": ["Earth", "Mars", "Venus", "Jupiter"],
                    "answer_index": 1,
                    "rationale": "Mars appears red because of iron oxide on its surface."
                },
                {
                    "question": "What is the chemical symbol for water?",
                    "choices": ["H2O", "O2", "HO", "H3O"],
                    "answer_index": 0,
                    "rationale": "Two hydrogens and one oxygen."
                },
                {
                    "question": "Who wrote 'Romeo and Juliet'?",
                    "choices": ["William Shakespeare", "Charles Dickens", "Leo Tolstoy", "Mark Twain"],
                    "answer_index": 0,
                    "rationale": "A Shakespeare tragedy from the late 16th century."
                },
                {
                    "question": "Which gas do plants absorb during photosynthesis?",
                    "choices": ["Oxygen", "Carbon Dioxide", "Nitrogen", "Hydrogen"],
                    "answer_index": 1,
                    "rationale": "Plants consume CO₂ and release O₂."
                },
                {
                    "question": "What is Earth’s largest ocean?",
                    "choices": ["Atlantic", "Indian", "Pacific", "Arctic"],
                    "answer_index": 2,
                    "rationale": "The Pacific covers about one-third of Earth’s surface."
                },
            ][:n]
        }

    def strip_fences(s: str) -> str:
        s = s.strip()
        s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
        return s.strip()

    def extract_first_object(s: str):
        start = s.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
        return None

    def try_load_json(raw: str):
        cleaned = strip_fences(raw)
        obj = extract_first_object(cleaned) or cleaned
        obj = re.sub(r",(\s*[}\]])", r"\1", obj)
        return json.loads(obj)

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
        "Each question MUST have exactly 4 short choices. Include a 0-based 'answer_index' "
        "and a one-sentence 'rationale'. "
        "Return ONLY valid JSON — no markdown, no backticks, no commentary — with this schema. If token limit is reached atleast try to give the valid json:\n"
        '{ "questions": [ { "question": "string", "choices": ["string","string","string","string"], "answer_index": 0, "rationale": "string" } ] }'
    )

    try:
        model = build_quiz_model(
            system_prompt,
            response_mime_type="application/json",
            max_output_tokens=1400,
            temperature=0.4,
        )
        resp = model.generate_content(
            "Use the context below only if helpful to make topical questions. "
            "If context is too short, create general questions for the subject. "
            "Return ONLY JSON.\n\n"
            f"Context:\n{ctx}"
        )
        raw = (getattr(resp, "text", None) or "").strip()

        try:
            payload = try_load_json(raw)
        except Exception as je:
            logging.warning("Quiz JSON parse failed: %s\nRaw: %s", je, raw[:1000])
            return jsonify({"questions": dummy_quiz(n)["questions"], "note": "fallback_dummy"}), 200

        qs = payload.get("questions", [])
        if not isinstance(qs, list) or not qs:
            return jsonify({"questions": dummy_quiz(n)["questions"], "note": "fallback_dummy_empty"}), 200

        payload["questions"] = qs[:n]
        return jsonify(payload), 200

    except Exception as e:
        msg = str(e)
        if "429" in msg or "quota" in msg.lower():
            return jsonify({
                "error": "Gemini API quota exceeded — using dummy quiz.",
                "questions": dummy_quiz(n)["questions"],
                "note": "fallback_quota"
            }), 200
    logging.exception("quiz_json fatal")
    return jsonify({
        "error": "Quiz generation failed — using dummy quiz.",
        "questions": dummy_quiz(n)["questions"],
        "note": "fallback_error"
    }), 200
