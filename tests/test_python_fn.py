import pytest
import time
from backend.app.tasks import python_fn

# -------- SUCCESS TEST --------
def test_python_fn_success():
    code_str = """
def user_func(x, y):
    return x + y
"""
    res = python_fn.run(func_code=code_str, func_args=[2, 3])
    assert res["success"]
    assert res["result"] == 5
    assert res["error"] is None

# -------- EXCEPTION TEST --------
def test_python_fn_exception():
    code_str = """
def user_func(x, y):
    return x / 0
"""
    res = python_fn.run(func_code=code_str, func_args=[2, 3])
    assert not res["success"]
    assert res["result"] is None
    assert "ZeroDivisionError" in res["error"]

# -------- TIMEOUT TEST --------
def test_python_fn_timeout():
    code_str = """
def user_func(x):
    import time
    time.sleep(10)
    return x
"""
    res = python_fn.run(func_code=code_str, func_args=[5])
    assert not res["success"]
    assert res["result"] is None
    assert "Timeout" in res["error"]

# -------- CUSTOM OPERATION TEST --------
def test_python_fn_custom_operation():
    code_str = """
def user_func(x, y):
    return x * y
"""
    res = python_fn.run(func_code=code_str, func_args=[4, 5])
    assert res["success"]
    assert res["result"] == 20
