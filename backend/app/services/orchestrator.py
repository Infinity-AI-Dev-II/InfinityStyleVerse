# backend/app/services/orchestrator.py
#   Provides a mock orchestration layer for AI/ML services.
#   - Generates a unique trace ID for each request.
#   - Routes requests to different mock services (VisionaryAI, StyleSense).
#   - Simulates processing latency and returns sample responses.
#   - Logs execution start, end, and elapsed time for observability.
#   - Can later be extended to call real HTTP/RPC services instead of mocks
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger("infinifybrain")

def _start_trace() -> str:
    return uuid.uuid4().hex

def call_visionaryai(payload: Dict[str, Any], trace_id:str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    logger.info("[trace=%s] call_visionaryai() start payload=%s", trace_id, payload)

    time.sleep(0.05)  # simulate latency
    image = payload.get("image_url") or payload.get("image_b64", "<no-image>")
    result = {
        "tags": ["dress", "floral", "summer"],
        "confidence": 0.92,
        "input_ref": image,
    }
    logger.info("[trace=%s] call_visionaryai() ok elapsed_ms=%.2f", trace_id, (time.perf_counter()-t0)*1000)
    return result

def call_stylesense(payload: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    logger.info("[trace=%s] call_stylesense() start payload=%s", trace_id, payload)
    # --- mock behavior ---
    time.sleep(0.05)
    attrs = payload.get("attributes", {})
    result = {
        "recommendations": [
            {"style": "boho", "score": 0.88},
            {"style": "minimal", "score": 0.76},
        ],
        "based_on": attrs,
    }
    logger.info("[trace=%s] call_stylesense() ok elapsed_ms=%.2f", trace_id, (time.perf_counter()-t0)*1000)
    return result

def route_to_service(service: str, payload: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
    service = (service or "").lower().strip()
    if service == "visionaryai":
        return call_visionaryai(payload, trace_id)
    if service == "stylesense":
        return call_stylesense(payload, trace_id)
    raise ValueError(f"Unknown service '{service}'. Valid: visionaryai, stylesense")

def orchestrate(request_data: Dict[str, Any], mock_mode: bool = True) -> Dict[str, Any]:
    """
    request_data: { "service": "visionaryai|stylesense", "payload": {...} }
    """
    trace_id = _start_trace()
    t0 = time.perf_counter()
    logger.info("[trace=%s] orchestrate() start data=%s", trace_id, request_data)

    service = request_data.get("service")
    payload = request_data.get("payload", {})

    if mock_mode:
        resp = route_to_service(service, payload, trace_id)
    else:
        # Placeholder for real HTTP/RPC calls in the future.
        resp = route_to_service(service, payload, trace_id)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    out = {
        "trace_id": trace_id,
        "service": service,
        "response": resp,
        "elapsed_ms": round(elapsed_ms, 2),
    }
    logger.info("[trace=%s] orchestrate() done elapsed_ms=%.2f", trace_id, elapsed_ms)
    return out