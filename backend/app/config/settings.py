import os

class Settings:
    # Flask
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")

    # Database
    DB_USER = os.getenv("POSTGRES_USER", "postgres")
    DB_PASS = os.getenv("POSTGRES_PASSWORD", "Nakshath7031")
    DB_HOST = os.getenv("DATABASE_HOST", "postgres")   # <-- use service name, not 127.0.0.1
    DB_PORT = os.getenv("DATABASE_PORT", "5432")
    DB_NAME = os.getenv("POSTGRES_DB", "infinitystyleversedb")

    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # JWT
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "super-secret-jwt-key")

    # Orchestrator
    ORCHESTRATOR_MOCK_MODE = True
    ORCHESTRATOR_LOG_LEVEL = "INFO"

    # HTTP Hardening
    HTTP_ALLOWED_HOSTS = os.getenv(
        "HTTP_ALLOWED_HOSTS",
        "api.example.com,auth.example.com"
    ).split(",")
    HTTP_ALLOWED_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    HTTP_REQUEST_TIMEOUT = 5.0

    # Rate Limiting
    RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", 60))

    # Celery/Redis
    REDIS_HOST = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT = os.getenv("REDIS_PORT", "6379")
    REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

    CELERY_BROKER_URL = REDIS_URL
    CELERY_RESULT_BACKEND = REDIS_URL


settings = Settings()
