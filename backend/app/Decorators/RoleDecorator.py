# backend/app/Decorators/decorators.py
#   Implements a Flask decorator to enforce role-based access control on routes.
from functools import wraps
from flask import redirect, url_for, flash
from flask_login import current_user

def role_required(required_role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                flash("Please log in to access this page.", "warning")
                return redirect(url_for('main.login_page'))
            if current_user.role != required_role:
                flash("You do not have permission to access this page.", "danger")
                return redirect(url_for('main.home'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator
