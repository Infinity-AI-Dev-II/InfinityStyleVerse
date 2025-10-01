#tests/test_auth.py
import pytest
from backend.app.models.user import User
from backend.app.models.roles import Role
from backend.app.database import db

# ---------------------------
# Fixture to seed test roles and users
# ---------------------------
@pytest.fixture(scope="function")
def seed_roles_and_users(app_ctx):
    """Seed roles and test users into DB"""
    # Clear existing users and roles
    User.query.delete()
    Role.query.delete()
    db.session.commit()

    # Create roles
    admin_role = Role(role_name="admin")
    user_role = Role(role_name="user")
    db.session.add_all([admin_role, user_role])
    db.session.commit()

    # Create users using password setter
    admin = User(
        name="Admin",
        email="admin@example.com"
    )
    admin.password = "Admin123!"  # uses setter to hash
    admin.role = admin_role

    alice = User(
        name="Alice",
        email="alice@example.com"
    )
    alice.password = "User123!"  # uses setter to hash
    alice.role = user_role

    db.session.add_all([admin, alice])
    db.session.commit()

    return admin, alice
# ---------------------------
# Helper to get user by email
# ---------------------------
@pytest.fixture
def get_user(app_ctx):
    def _get_user(email):
        return User.query.filter_by(email=email).first()
    return _get_user


# ---------------------------
# Helper to login and get tokens (access, refresh, csrf)
# ---------------------------
@pytest.fixture
def login_and_get_tokens(client):
    def _login(email, password):
        resp = client.post(
            "/auth/login",
            json={"email": email, "password": password}
        )
        data = resp.get_json()
        access = data.get("access_token")
        refresh = data.get("refresh_token")
        csrf = None  # CSRF header not needed unless cookies are used
        return access, refresh, csrf
    return _login


# ---------------------------
# Test registration
# ---------------------------
def test_register_success(client, app_ctx, seed_roles_and_users, get_user):
    resp = client.post(
        "/auth/register",
        json={
            "name": "Bob",
            "email": "bob@example.com",
            "password": "Bob123!",
            "role": "user"
        }
    )
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["msg"] == "User registered successfully"

    user = get_user("bob@example.com")
    assert user is not None
    assert user.verify_password("Bob123!")


# ---------------------------
# Test login
# ---------------------------
def test_login_and_tokens(client, app_ctx, seed_roles_and_users, get_user, login_and_get_tokens):
    access, refresh, csrf = login_and_get_tokens("alice@example.com", "User123!")
    assert access is not None
    assert refresh is not None

    user = get_user("alice@example.com")
    assert user is not None
    assert user.verify_password("User123!")


# ---------------------------
# Test refresh token
# ---------------------------
def test_refresh_token_flow(client, app_ctx, seed_roles_and_users, login_and_get_tokens):
    access, refresh, csrf = login_and_get_tokens("alice@example.com", "User123!")

    resp = client.post(
        "/auth/refresh",
        headers={"Authorization": f"Bearer {refresh}"}
    )
    assert resp.status_code == 200

    new_access = resp.get_json().get("access_token")
    assert new_access is not None
    assert new_access != access
