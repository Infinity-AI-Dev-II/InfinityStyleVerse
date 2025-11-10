# backend/app/tasks/http_call.py
#   Provides Celery tasks for parallel ("fan-out") and aggregation ("fan-in") of workflow steps.
#   - `map_runner`: executes a user-defined function on a list of items concurrently using threads.
#       • Captures individual results and errors for each item.
#   - `join_runner`: aggregates outputs from multiple parallel steps and determines the overall status.
#   - Enables scalable execution of independent workflow steps and collection of their results.

import urllib.parse as urlparse
import requests
import logging
from backend.celery_app import celery_app
from backend.app.utils.http_client import request as safe_request
from backend.app.utils.rate_limit import allow
from backend.app.config.settings import settings
from backend.app.tasks.utils import safe_retry
from celery import shared_task
from backend.app.tasks.decorators import instrument

logger = logging.getLogger(__name__)

@celery_app.task(
    bind=True,
    name="backend.app.tasks.http_call.http_call_task",
    autoretry_for=(requests.RequestException,),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=3,
    time_limit=30,
    soft_time_limit=25,
    queue="io",
)
def http_call_task(self, url: str, method: str = "GET", headers: dict | None = None,
                   json: dict | None = None, access_token: str | None = None, timeout: int | None = None):
    """
    Execute an HTTP request safely with retry/backoff, timeout, and rate limiting.
    Returns standardized JSON/dict: {status, data, error}.
    """
    headers = headers or {}
    timeout = timeout or settings.HTTP_REQUEST_TIMEOUT

    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    if not url:
        return {"status": None, "data": None, "error": "URL cannot be None"}

    parsed = urlparse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return {"status": None, "data": None, "error": f"Invalid URL, host missing: {url}"}

    if not allow(host):
        return {"status": None, "data": None, "error": f"Rate limit exceeded for host: {host}"}

    try:
        resp = safe_request(
            method=method,
            url=url,
            headers=headers,
            timeout=timeout,
            **({"json": json} if json else {})
        )

        # Handle non-JSON gracefully
        content_type = resp.headers.get("content-type", "")
        try:
            data = resp.json() if "application/json" in content_type else resp.text[:200]
        except Exception:
            data = resp.text[:200]

        # Handle 4xx/5xx responses gracefully
        if 400 <= resp.status_code < 600:
            logger.warning(f"HTTP call returned error status {resp.status_code}: {url}")
            return {"status": resp.status_code, "data": data, "error": f"HTTP error {resp.status_code}"}

        logger.info(f"HTTP call successful: {method} {url} status={resp.status_code}")
        return {"status": resp.status_code, "data": data, "error": None}

    except requests.RequestException as exc:
        logger.error(f"HTTP call failed: {exc} | URL: {url}")
        return safe_retry(self, exc)


@shared_task(name="tasks.http_call.run")
@instrument("http_call")
def http_call(url: str, method: str = "GET", **kwargs):
    """
    Wrapper task for HTTP calls. Returns standardized JSON/dict.
    """
    from backend.app.utils import http_client

    if not url:
        return {"status": None, "data": None, "error": "URL cannot be None"}

    try:
        response = http_client.request(method, url, **kwargs)
        content_type = response.headers.get("content-type", "")
        try:
            data = response.json() if "application/json" in content_type else response.text[:200]
        except Exception:
            data = response.text[:200]

        if 400 <= response.status_code < 600:
            return {"status": response.status_code, "data": data, "error": f"HTTP error {response.status_code}"}

        return {"status": response.status_code, "data": data, "error": None}

    except Exception as e:
        logger.error(f"HTTP wrapper task failed: {e} | URL: {url}")
        return {"status": None, "data": None, "error": str(e)}
