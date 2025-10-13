from datetime import datetime
from extensions import db

class Flashcard(db.Model):
    __tablename__ = "flashcard"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    subject = db.Column(db.String(64), index=True)
    front = db.Column(db.Text, nullable=False)
    back = db.Column(db.Text, nullable=False)

    repetitions = db.Column(db.Integer, default=0)
    ease = db.Column(db.Float, default=2.5)
    interval = db.Column(db.Integer, default=0)      # days
    due_at = db.Column(db.DateTime, default=datetime.utcnow)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ReviewLog(db.Model):
    __tablename__ = "review_log"
    id = db.Column(db.Integer, primary_key=True)
    card_id = db.Column(db.Integer, db.ForeignKey("flashcard.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    quality = db.Column(db.Integer, nullable=False)  # 0..5
    reviewed_at = db.Column(db.DateTime, default=datetime.utcnow)
    interval_after = db.Column(db.Integer)
    ease_after = db.Column(db.Float)
    repetitions_after = db.Column(db.Integer)
