from flask import Blueprint, render_template, request

from models.subject import SUBJECTS

chat_bp = Blueprint("chat", __name__)


@chat_bp.route('/')
def index():
    return render_template('index.html')


@chat_bp.route('/chat')
def chat():
    subject = request.args.get('subject', 'math')
    if subject not in SUBJECTS:
        subject = 'math'
    return render_template('chat.html', subject=subject, subject_name=SUBJECTS[subject])
