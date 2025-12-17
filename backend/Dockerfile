# backend/Dockerfile
#Builds a Docker image for the backend Flask/Celery application
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# system deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential gcc libpq-dev \
       curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# create app user
RUN useradd --create-home appuser
WORKDIR /app

# copy requirements first for caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install Flask-Caching

# upgrade pip once and install all deps (incl. boto3) from requirements.txt
# RUN python -m pip install --upgrade pip \
#     && pip install --no-cache-dir -r /app/requirements.txt

# RUN --mount=type=cache,target=/root/.cache/pip \
#     pip install -r requirements.txt

# copy application code
COPY . /app

# ensure scripts are executable
RUN chmod +x /app/entrypoint.sh

# use non-root
USER appuser

ENV FLASK_ENV=production
ENV PYTHONPATH=/app

EXPOSE 8000

# entrypoint -> run migrations (if any) and start gunicorn
ENTRYPOINT ["/app/entrypoint.sh"]
# CMD ["gunicorn", "backend.app:create_app()", "-b", "0.0.0.0:8000", "--workers", "4", "--worker-class", "gthread", "--threads", "4"]
CMD ["gunicorn", "backend.app:create_app()", "-b", "0.0.0.0:8000", "--workers", "4", "--worker-class", "gevent", "--threads", "4", "--timeout", "300"]
