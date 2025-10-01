import pytest
import requests
from unittest.mock import patch
from celery.exceptions import MaxRetriesExceededError
from backend.app.tasks import model_call, join_map, python_fn

# -------- MODEL CALL TESTS --------
@patch("backend.app.tasks.model_call.requests.Session.post")
@patch("backend.app.tasks.model_call.model_call_task.retry")
def test_model_call_success(mock_retry, mock_post):
    mock_post.return_value.json.return_value = {"prediction": 1}
    mock_post.return_value.status_code = 200

    # retry should not interfere
    mock_retry.side_effect = MaxRetriesExceededError()

    res = model_call.model_call_task.run("http://mlapi", {"input": 1})
    assert res["success"]
    assert res["output"]["prediction"] == 1

@patch("backend.app.tasks.model_call.requests.Session.post")
@patch("backend.app.tasks.model_call.model_call_task.retry")
def test_model_call_failure(mock_retry, mock_post):
    # simulate request failure
    mock_post.side_effect = requests.RequestException("fail")

    # patch retry to raise MaxRetriesExceededError so task returns error dict
    mock_retry.side_effect = MaxRetriesExceededError()

    res = model_call.model_call_task.run("http://mlapi", {"input": 1})
    assert res["success"] is False
    assert res["output"] is None
    assert "fail" in res["error"]

# -------- JOIN/MAP TESTS --------
def test_join_task():
    result = join_map.join_task.run(1, 2, None, 3)
    assert result["success"]
    assert result["output"] == [1, 2, 3]

def test_map_task():
    code_str = "def user_func(x): return x*2"
    result = join_map.map_task.run(code_str, [1, 2, 3])
    assert result["success"]
    assert result["output"] == [2, 4, 6]

# -------- PYTHON_FN TESTS --------
def test_python_fn_add():
    code_str = """
def user_func(a, b):
    return a + b
"""
    res = python_fn.run(func_code=code_str, func_args=[2, 3])
    assert res["success"]
    assert res["result"] == 5
