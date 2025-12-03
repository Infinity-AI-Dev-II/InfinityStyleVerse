#validate the scopes along side with the JWT token in the request
from flask_jwt_extended import verify_jwt_in_request, get_jwt
from functools import wraps
from flask import abort
def require_scopes(required_scopes):
    def wrapper(fn):
        @wraps(fn)
        def decorator(*args, **kwargs):
            verify_jwt_in_request() # ensure JWT is valid
            claims = get_jwt() #JWT data with @jwt_required part
            mainScopes = claims.get("scopes", [])
            for scopes in required_scopes:
                if scopes not in mainScopes:
                    abort(403, description="Forbidden: Insufficient scopes")
            return fn(*args, **kwargs)
        return decorator
    return wrapper
            




