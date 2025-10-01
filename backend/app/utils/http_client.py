# backend/app/utils/http_client.py
import urllib.parse as urlparse
import logging
import requests
from backend.app.config.settings import settings

# Headers considered sensitive that should be redacted
SENSITIVE_HEADERS = {"authorization", "x-api-key", "cookie"}

def _safe_headers(headers: dict | None) -> dict:
    """
    Redact sensitive headers before logging.
    """
    if not headers:
        return {}
    redacted = {}
    for k, v in headers.items():
        if k.lower() in SENSITIVE_HEADERS:
            redacted[k] = "***REDACTED***"
        else:
            redacted[k] = v
    return redacted

def check_allowlist(method: str, url: str):
    """
    Allow requests only to approved hosts and HTTP methods.
    """
    host = (urlparse.urlparse(url).hostname or "").lower()
    method = method.upper()
    allowed_hosts = [h.strip().lower() for h in settings.HTTP_ALLOWED_HOSTS]
    allowed_methods = [m.strip().upper() for m in settings.HTTP_ALLOWED_METHODS]

    if host not in allowed_hosts:
        raise ValueError(f"Host '{host}' not allowed")
    if method not in allowed_methods:
        raise ValueError(f"Method '{method}' not allowed")

def request(method: str, url: str, headers: dict | None = None, **kwargs) -> requests.Response:
    """
    Make a safe HTTP request after checking allowlist.
    """
    check_allowlist(method, url)
    safe_hdrs = _safe_headers(headers)
    logging.info(
        "http_request",
        extra={"method": method, "url": url, "headers": safe_hdrs}
    )

    timeout = kwargs.pop("timeout", settings.HTTP_REQUEST_TIMEOUT)
    # Pass headers as empty dict if None
    return requests.request(method=method, url=url, headers=headers or {}, timeout=timeout, **kwargs)
print("Allowed hosts:", settings.HTTP_ALLOWED_HOSTS)
