from functools import wraps
from flask import jsonify


def validate_request_size(request: any, max_json_kb=500):
    """Decorator to validate JSON payload size"""

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            content_length = request.content_length
            if content_length and content_length > max_json_kb * 1024:
                return (
                    jsonify(
                        {
                            "error": "REQUEST_TOO_LARGE",
                            "message": f"JSON payload exceeds {max_json_kb}KB",
                            "received_kb": content_length / 1024,
                        }
                    ),
                    413,
                )
            return f(*args, **kwargs)

        return wrapper

    return decorator




