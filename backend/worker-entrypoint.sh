#!/bin/bash
set -e

# Wait for Redis + Postgres (both may be needed)
echo "=> waiting for redis at ${REDIS_HOST}:${REDIS_PORT} ..."
RETRIES=30
until nc -z "$REDIS_HOST" "$REDIS_PORT"; do
  >&2 echo "Redis is unavailable - sleeping"
  sleep 1
  RETRIES=$((RETRIES-1))
  if [ $RETRIES -le 0 ]; then
    >&2 echo "Redis did not become available in time"
    exit 1
  fi
done

echo "Redis available"

# Ensure DB is up too (if your tasks use DB)
echo "=> waiting for postgres at ${DATABASE_HOST}:${DATABASE_PORT} ..."
RETRIES=30
until nc -z "$DATABASE_HOST" "$DATABASE_PORT"; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 1
  RETRIES=$((RETRIES-1))
  if [ $RETRIES -le 0 ]; then
    >&2 echo "Postgres did not become available in time"
    exit 1
  fi
done

echo "Starting celery worker..."
exec "$@"
