from flask import Blueprint, render_template, request, redirect, url_for, session
from flask_login import current_user

chat_bp = Blueprint("chat", __name__)


def login_or_guest_required(view):
    from functools import wraps
    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_user.is_authenticated or session.get("guest") is True:
            return view(*args, **kwargs)
        return redirect(url_for("chat.landing"))

    return wrapped


@chat_bp.route("/")
def landing():
    if current_user.is_authenticated or session.get("guest"):
        return redirect(url_for("chat.subjects"))
    return render_template("landing.html")


@chat_bp.post("/guest-start")
def guest_start():
    session["guest"] = True
    return redirect(url_for("chat.subjects"))


@chat_bp.get("/guest-end")
def guest_end():
    session.pop("guest", None)
    return redirect(url_for("chat.landing"))


@chat_bp.route("/subjects")
def subjects():
    return render_template("index.html")


@chat_bp.route("/chat")
@login_or_guest_required
def chat():
    subject = request.args.get("subject", "math")
    subject_map = {
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
    subject_name = subject_map.get(subject, subject.title())
    return render_template("chat.html", subject=subject, subject_name=subject_name)
