# backend/app/otel.py
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.celery import CeleryInstrumentor
import logging

def init_tracing(app=None, celery_app=None, otlp_endpoint: str | None = None):
    resource = Resource.create({"service.name": "echoos-workflow"})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    exporter = OTLPSpanExporter(endpoint=otlp_endpoint or "http://localhost:4317")
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    if app is not None:
        FlaskInstrumentor().instrument_app(app)
        logging.getLogger(__name__).info("Flask instrumented")
    if celery_app is not None:
        CeleryInstrumentor().instrument()
        logging.getLogger(__name__).info("Celery instrumented")
