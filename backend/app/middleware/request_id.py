import uuid
from flask import g, request

def register_request_id_middleware(app):
    @app.before_request
    def add_request_id():
        rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        g.request_id = rid

    @app.after_request
    def add_response_header(response):
        response.headers["X-Request-ID"] = getattr(g, "request_id", None)
        return response
