#!/bin/bash
set -e

# Wait for Postgres to be available
host="$DATABASE_HOST"
port="$DATABASE_PORT"

echo "=> waiting for postgres at ${host}:${port} ..."
RETRIES=30
until nc -z "$host" "$port"; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 1
  RETRIES=$((RETRIES-1))
  if [ $RETRIES -le 0 ]; then
    >&2 echo "Postgres did not become available in time"
    exit 1
  fi
done

echo "Postgres is up - continuing"

# Optional: run DB migrations or create tables
# If you use Flask-Migrate / alembic, you could run: flask db upgrade
# Keep this simple: if you want to run a custom migrate script, place it here
if [ -n "$FLASK_MIGRATE" ]; then
  echo "Running flask db upgrade"
  flask db upgrade || true
fi

# start the passed CMD (gunicorn) or fallback to flask run
exec "$@"
