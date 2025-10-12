from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required
from flask_bcrypt import Bcrypt
from models.user import User
from extensions import db

auth_bp = Blueprint("auth", __name__)
bcrypt = Bcrypt()


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        pw = request.form["password"]
        if User.query.filter_by(email=email).first():
            flash("Email already registered!")
            return redirect(url_for("auth.register"))
        hashed_pw = bcrypt.generate_password_hash(pw).decode("utf-8")
        user = User(email=email, password=hashed_pw, name=request.form.get("name"))
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for("chat.index"))
    return render_template("register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        pw = request.form["password"]
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, pw):
            login_user(user)
            return redirect(url_for("chat.index"))
        flash("Invalid email or password")
        return redirect(url_for("auth.login"))
    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))
