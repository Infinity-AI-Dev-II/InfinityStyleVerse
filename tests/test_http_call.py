import pytest
import requests
from unittest.mock import patch, MagicMock
from backend.app.tasks.http_call import http_call_task, http_call

# Helper to create mock responses
def mock_response(status=200, json_data=None, text_data=None, headers=None):
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.headers = headers or {"content-type": "application/json"}
    if json_data is not None:
        mock_resp.json.return_value = json_data
    if text_data is not None:
        mock_resp.text = text_data
    return mock_resp

# -------- SUCCESS TESTS --------
@patch("backend.app.tasks.http_call.allow", return_value=True)
@patch("backend.app.tasks.http_call.safe_request")
def test_http_call_allowed(mock_safe_request, mock_allow):
    mock_safe_request.return_value = mock_response(json_data={"ok": True}, status=200)
    res = http_call_task.run("https://httpbin.org/get", "GET")
    assert res["status"] == 200
    assert res["data"]["ok"] is True
    assert res["error"] is None

# -------- RATE LIMIT BLOCK --------
@patch("backend.app.tasks.http_call.allow", return_value=False)
def test_http_call_blocked(mock_allow):
    res = http_call_task.run("https://httpbin.org/get", "GET")
    assert res["status"] is None
    assert res["data"] is None
    assert "Rate limit exceeded" in res["error"]

# -------- REQUEST EXCEPTION & RETRY --------
@patch("backend.app.tasks.http_call.allow", return_value=True)
@patch("backend.app.tasks.http_call.safe_request")
@patch("backend.app.tasks.http_call.http_call_task.retry")
def test_http_call_request_exception(mock_retry, mock_safe_request, mock_allow):
    mock_safe_request.side_effect = requests.RequestException("fail")
    # patch retry to raise MaxRetriesExceededError so task returns error dict
    from celery.exceptions import MaxRetriesExceededError
    mock_retry.side_effect = MaxRetriesExceededError()

    res = http_call_task.run("https://httpbin.org/get", "GET")
    assert res["status"] is None
    assert res["data"] is None
    assert "fail" in res["error"]

# -------- HTTP WRAPPER TEST --------
@patch("backend.app.utils.http_client.request")
def test_http_call_wrapper_success(mock_http_client):
    mock_http_client.return_value = mock_response(json_data={"ok": True}, status=200)
    res = http_call("https://httpbin.org/get")
    assert res["status"] == 200
    assert res["data"]["ok"] is True
    assert res["error"] is None
