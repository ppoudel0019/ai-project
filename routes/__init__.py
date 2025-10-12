from .auth import auth_bp
from .ai import ai_bp
from .chat import chat_bp
from .flashcard import flashcards_bp
from .profile import profile_bp


def register_blueprints(app):
    app.register_blueprint(chat_bp)
    app.register_blueprint(ai_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(flashcards_bp)
    app.register_blueprint(profile_bp)