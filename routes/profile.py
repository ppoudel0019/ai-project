from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from extensions import db, bcrypt

profile_bp = Blueprint("profile", __name__, url_prefix="/profile")


@profile_bp.get("")
@login_required
def view_profile():
    return render_template("profile.html", user=current_user)


@profile_bp.post("")
@login_required
def update_profile():
    name = request.form.get("name", "").strip()
    bio = request.form.get("bio", "").strip()
    tz = request.form.get("timezone", "").strip() or current_user.timezone
    fav = request.form.get("favorite_subject", "").strip() or current_user.favorite_subject
    if len(name) > 100:
        flash("Name is too long.")
        return redirect(url_for("profile.view_profile"))
    current_user.name = name or current_user.name
    current_user.bio = bio
    current_user.timezone = tz
    current_user.favorite_subject = fav
    db.session.commit()
    flash("Profile updated.")
    return redirect(url_for("profile.view_profile"))


@profile_bp.post("/change-password")
@login_required
def change_password():
    current_pw = request.form.get("current_password", "")
    new_pw = request.form.get("new_password", "")
    confirm = request.form.get("confirm_password", "")
    if not bcrypt.check_password_hash(current_user.password, current_pw):
        flash("Current password is incorrect.")
        return redirect(url_for("profile.view_profile"))
    if len(new_pw) < 8:
        flash("New password must be at least 8 characters.")
        return redirect(url_for("profile.view_profile"))
    if new_pw != confirm:
        flash("Passwords do not match.")
        return redirect(url_for("profile.view_profile"))
    current_user.password = bcrypt.generate_password_hash(new_pw).decode("utf-8")
    db.session.commit()
    flash("Password changed.")
    return redirect(url_for("profile.view_profile"))
