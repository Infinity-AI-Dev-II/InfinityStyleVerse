import json
import os
import sys

import pytest
from flask import Flask
from flask_jwt_extended import JWTManager, create_access_token

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
sys.path.append(os.path.abspath("."))

from backend.app.StyleSense.StyleAPI.BodyMorphRoutes import bodyMorph_bp  # noqa: E402
from backend.app.services.redisKeyGenerate import generate_image_cache_key  # noqa: E402

SIGNED_URL = "https://example.com/image.jpg?X-Amz-Signature=abc123"


class DummyRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value


@pytest.fixture
def app(monkeypatch):
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["JWT_SECRET_KEY"] = "test-secret"
    JWTManager(app)

    # patch network/storage dependencies
    monkeypatch.setattr(
        "backend.app.StyleSense.StyleAPI.BodyMorphRoutes.load_image_from_signed_url",
        lambda url: b"image-bytes",
    )

    idempotency_store = {}

    class FakeRow:
        def __init__(self, record):
            self.response_json = record["response_json"]
            self.request_hash = record.get("request_hash")

    def fake_read(key):
        record = idempotency_store.get(key)
        return FakeRow(record) if record else None

    def fake_write(key, request_hash, response_json):
        idempotency_store[key] = {
            "request_hash": request_hash,
            "response_json": json.dumps(response_json),
        }

    monkeypatch.setattr("backend.app.StyleSense.StyleAPI.BodyMorphRoutes.read_idempotency", fake_read)
    monkeypatch.setattr("backend.app.StyleSense.StyleAPI.BodyMorphRoutes.write_idempotency", fake_write)

    app.config["REDIS_CLIENT"] = DummyRedis()
    app.register_blueprint(bodyMorph_bp)
    return app


@pytest.fixture
def auth_header(app):
    with app.app_context():
        token = create_access_token(identity="tester")
    return {"Authorization": f"Bearer {token}"}


def test_body_profile_happy_path(app, auth_header):
    client = app.test_client()
    resp = client.post(
        "/stylesense/body_profile",
        json={"image_uri": SIGNED_URL, "hints": {"height_cm": 168}},
        headers={**auth_header, "Idempotency-Key": "happy-1"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "body_type" in data
    assert "normalized" in data
    assert "request_id" in data


def test_body_profile_missing_image_uri(app, auth_header):
    client = app.test_client()
    resp = client.post(
        "/stylesense/body_profile",
        json={"hints": {}},
        headers={**auth_header, "Idempotency-Key": "missing-image"},
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"]["code"] == "INVALID_ARGUMENT"
    assert data["error"]["details"]["field"] == "image_uri"


def test_idempotency_returns_same_response(app, auth_header):
    client = app.test_client()
    headers = {**auth_header, "Idempotency-Key": "idem-test"}
    payload = {"image_uri": SIGNED_URL, "hints": {}}

    first = client.post("/stylesense/body_profile", json=payload, headers=headers)
    second = client.post("/stylesense/body_profile", json=payload, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.get_json() == second.get_json()


def test_cache_hit_skips_engine(app, auth_header, monkeypatch):
    # Prime cache
    cache = app.config["REDIS_CLIENT"]
    cached_response = {
        "request_id": "req_cached",
        "body_type": "cached",
        "confidence": 0.9,
        "normalized": {},
        "trace": [],
    }
    hints = {"shape_hint": "cache-me"}
    cache_key = generate_image_cache_key(b"image-bytes", hints)
    cache.set(cache_key, json.dumps(cached_response))

    class ExplodingEngine:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("Engine should not run on cache hit")

    monkeypatch.setattr("backend.app.StyleSense.StyleAPI.BodyMorphRoutes.BodyMorphEngine", ExplodingEngine)

    client = app.test_client()
    resp = client.post(
        "/stylesense/body_profile",
        json={"image_uri": SIGNED_URL, "hints": hints},
        headers={**auth_header, "Idempotency-Key": "cache-hit"},
    )

    assert resp.status_code == 200
    assert resp.get_json() == cached_response
