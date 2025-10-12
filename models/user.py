from flask_login import UserMixin
from datetime import datetime
from extensions import db


class User(db.Model, UserMixin):
    __tablename__ = 'user'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password = db.Column(db.String(128), nullable=False)
    name = db.Column(db.String(100))

    bio = db.Column(db.Text, default="")
    timezone = db.Column(db.String(64), default="America/Chicago")
    favorite_subject = db.Column(db.String(64), default="math")
    avatar_url = db.Column(db.String(512))

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
