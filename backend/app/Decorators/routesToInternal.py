#decorator to make a toute internal-only
from flask import abort, request
from sklearn.conftest import wraps

ALLOWED_IP = ['127.0.0.1']
def limmit_route_internal(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        clientIP = request.remote_addr
        if clientIP not in ALLOWED_IP:
            abort(403)
        return func(*args, **kwargs)
    return wrapper

    
