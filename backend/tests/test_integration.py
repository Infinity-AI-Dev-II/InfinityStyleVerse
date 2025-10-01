import pytest
from flask_jwt_extended import create_access_token, create_refresh_token
from backend.app.models import User

@pytest.fixture
def login_and_get_tokens(seed_roles_and_users):
    def _login(email, password):
        user = User.query.filter_by(email=email).first()
        assert user is not None
        assert user.verify_password(password)  # Make sure password is correct

        #  JWT identity must be a string
        access_token = create_access_token(identity=str(user.id))
        refresh_token = create_refresh_token(identity=str(user.id))
        return access_token, refresh_token
    return _login

def test_login_then_access_admin_dashboard(client, seed_roles_and_users, login_and_get_tokens):
    admin, _ = seed_roles_and_users
    access, _ = login_and_get_tokens(admin.email, "Admin123!")
    
    headers = {
        "Authorization": f"Bearer {access}",
        "Content-Type": "application/json"
    }
    resp = client.get("/admin/users", headers=headers)
    assert resp.status_code == 200
