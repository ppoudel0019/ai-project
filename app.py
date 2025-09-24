from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
import os

# Load environment variables
load_dotenv()

# ---- Gemini setup ----
# Expect GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment.")

genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)
CORS(app)

# Store conversation history per session (volatile, in-memory)
# Your format: [{"role":"user"|"assistant", "content":"..."}]
conversations = {}

# Subject definitions
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
    """
    Gemini 1.5 supports a system instruction at model construction time.
    """
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system_prompt,
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 500,
        },
    )


def to_gemini_history(history):
    """
    Convert your stored history format (OpenAI-style role/content)
    to Gemini's expected format: [{"role":"user"|"model", "parts":[str]}]
    """
    converted = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "assistant":
            converted.append({"role": "model", "parts": [content]})
        else:
            # treat everything else as user (your code only uses "user" and "assistant")
            converted.append({"role": "user", "parts": [content]})
    return converted


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat')
def chat():
    subject = request.args.get('subject', 'math')
    if subject not in SUBJECTS:
        subject = 'math'
    return render_template('chat.html', subject=subject, subject_name=SUBJECTS[subject])


@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')
    subject = data.get('subject', 'math')
    session_id = data.get('session_id', 'default')

    if not question.strip():
        return jsonify({'error': 'Question cannot be empty'}), 400

    if subject not in SUBJECTS:
        subject = 'math'

    # init session history
    if session_id not in conversations:
        conversations[session_id] = []

    # last 4 Q&A pairs (8 messages)
    history = conversations[session_id][-8:] if len(conversations[session_id]) > 8 else conversations[session_id]

    try:
        system_prompt = (
            f"You are a helpful study assistant specializing in {SUBJECTS[subject]}. "
            f"Answer questions clearly and educationally. Keep responses concise but informative. "
            f"Focus only on {SUBJECTS[subject]} topics."
        )

        # Build Gemini model with system instruction
        model = build_gemini_model(system_prompt)

        # Start chat with converted history
        chat = model.start_chat(history=to_gemini_history(history))

        # Send the new user question
        gemini_response = chat.send_message(question)

        # Extract text answer
        answer = (gemini_response.text or "").strip()

        # Update our own stored history (still OpenAI-style for your app)
        conversations[session_id].append({"role": "user", "content": question})
        conversations[session_id].append({"role": "assistant", "content": answer})

        # Trim to last 8 messages
        if len(conversations[session_id]) > 8:
            conversations[session_id] = conversations[session_id][-8:]

        return jsonify({
            'answer': answer,
            'question': question,
            'subject': subject
        })

    except Exception as e:
        return jsonify({'error': f'Error generating response: {str(e)}'}), 500


@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    data = request.get_json()
    session_id = data.get('session_id', 'default')
    if session_id in conversations:
        conversations[session_id] = []
    return jsonify({'message': 'Conversation reset successfully'})


if __name__ == '__main__':
    # In production, use a proper WSGI/ASGI server and set debug=False
    app.run(debug=False, host='0.0.0.0', port=5001)
