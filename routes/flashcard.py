from datetime import datetime, timedelta

from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from sqlalchemy.sql import func

from extensions import db
from models.flashcard import Flashcard, ReviewLog

flashcards_bp = Blueprint("flashcards", __name__, url_prefix="/flashcards")

SUBJECT_CHOICES = ["math", "biology", "chemistry", "physics", "history", "english", "spanish", "french", "german",
                   "chinese"]


def _user_subjects():
    rows = db.session.query(Flashcard.subject) \
        .filter(Flashcard.user_id == current_user.id) \
        .distinct().all()
    return [r[0] for r in rows if r[0]]


def sm2_update(card: Flashcard, quality: int):
    q = max(0, min(5, quality))
    if q < 3:
        card.repetitions = 0
        card.interval = 1
    else:
        if card.repetitions == 0:
            card.interval = 1
        elif card.repetitions == 1:
            card.interval = 6
        else:
            card.interval = int(round(card.interval * card.ease))
        card.repetitions += 1
    card.ease = card.ease + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    if card.ease < 1.3:
        card.ease = 1.3
    card.due_at = datetime.utcnow() + timedelta(days=card.interval)
    return card


def get_next_card(subject: str, replay: bool):
    base = Flashcard.query.filter(Flashcard.user_id == current_user.id)
    if subject:
        base = base.filter(Flashcard.subject == subject)
    if replay:
        return base.order_by(func.random()).first()
    now = datetime.utcnow()
    return base.filter(Flashcard.due_at <= now).order_by(Flashcard.due_at.asc()).first()


def serialize_card(card: Flashcard):
    if not card:
        return None
    return {
        "id": card.id,
        "front": card.front,
        "back": card.back,
        "subject": card.subject or "general"
    }


@flashcards_bp.get("")
@login_required
def list_cards():
    subject = request.args.get("subject") or ""
    q = Flashcard.query.filter_by(user_id=current_user.id)
    if subject:
        q = q.filter(Flashcard.subject == subject)
    cards = q.order_by(Flashcard.updated_at.desc()).all()
    subjects = sorted(set(SUBJECT_CHOICES + _user_subjects()))
    return render_template("flashcards_list.html", cards=cards, subject=subject, subjects=subjects)


@flashcards_bp.get("/review")
@login_required
def review_queue():
    """
    Normal mode: shows next due card (due_at <= now).
    Replay mode: ?replay=1 -> ignore due dates; pick a random card from the (filtered) set.
    If ?partial=1 -> return JSON { has_card: bool, card, subject, replay }
    """
    subject = request.args.get("subject") or ""
    replay = (request.args.get("replay") or "").lower() in ("1", "true", "yes", "all")
    partial = (request.args.get("partial") or "").lower() in ("1", "true", "yes")

    card = get_next_card(subject, replay)
    if partial:
        return jsonify({
            "has_card": bool(card),
            "card": serialize_card(card),
            "subject": subject,
            "replay": replay
        })

    subjects = sorted(set(SUBJECT_CHOICES + _user_subjects()))
    return render_template("flashcards_review.html",
                           card=card, subject=subject, subjects=subjects, replay=replay)


@flashcards_bp.post("/create")
@login_required
def create_card():
    data = request.get_json(silent=True) or request.form
    front = (data.get("front") or "").strip()
    back = (data.get("back") or "").strip()
    subject = (data.get("subject") or "general").strip()
    if not front or not back:
        return jsonify({"error": "front and back are required"}), 400
    card = Flashcard(
        user_id=current_user.id,
        subject=subject or "general",
        front=front,
        back=back,
        repetitions=0,
        ease=2.5,
        interval=0,
        due_at=datetime.utcnow(),
    )
    db.session.add(card)
    db.session.commit()
    return jsonify({"ok": True, "id": card.id})


@flashcards_bp.post("/grade")
@login_required
def grade_card():
    data = request.get_json(silent=True) or request.form
    card_id = int(data.get("card_id", 0))
    quality = int(data.get("quality", 3))
    subject = (data.get("subject") or "").strip()
    replay = str(data.get("replay") or "").lower() in ("1", "true", "yes", "all")

    card = Flashcard.query.filter_by(id=card_id, user_id=current_user.id).first()
    if not card:
        return jsonify({"error": "Card not found"}), 404

    sm2_update(card, quality)
    db.session.add(ReviewLog(
        card_id=card.id, user_id=current_user.id, quality=quality,
        interval_after=card.interval, ease_after=card.ease, repetitions_after=card.repetitions
    ))
    db.session.commit()

    # Immediately return the next card so the UI can move on without reloading
    next_card = get_next_card(subject, replay)
    return jsonify({
        "ok": True,
        "next": {
            "has_card": bool(next_card),
            "card": serialize_card(next_card)
        }
    })


@flashcards_bp.post("/delete")
@login_required
def delete_card():
    data = request.get_json(silent=True) or request.form
    card_id = int(data.get("card_id", 0))
    subject = (data.get("subject") or "").strip()
    replay = str(data.get("replay") or "").lower() in ("1", "true", "yes", "all")

    card = Flashcard.query.filter_by(id=card_id, user_id=current_user.id).first()
    if not card:
        return jsonify({"error": "Card not found"}), 404

    ReviewLog.query.filter_by(card_id=card.id, user_id=current_user.id).delete()
    db.session.delete(card)
    db.session.commit()

    next_card = get_next_card(subject, replay)
    return jsonify({
        "ok": True,
        "next": {
            "has_card": bool(next_card),
            "card": serialize_card(next_card)
        }
    })
