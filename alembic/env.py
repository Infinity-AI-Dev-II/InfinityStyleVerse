import sys
import os
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Alembic Config object
config = context.config

# Logging configuration
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# --- Add project paths so Python can find app and modules ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
APP_PATH = os.path.join(PROJECT_ROOT, "app")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, APP_PATH)

# --- Imports ---
try:
    from config.settings import Config  # backend/app/config.py
    from models import Base    # backend/app/models/__init__.py
except ImportError as e:
    print("Error importing Config or Base:", e)
    raise

# Alembic target metadata
target_metadata = Base.metadata

# Override SQLAlchemy URL
config.set_main_option("sqlalchemy.url", Config.SQLALCHEMY_DATABASE_URI)


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"}
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
