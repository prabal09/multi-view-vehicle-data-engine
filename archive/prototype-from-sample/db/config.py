"""Database connection helper shared by the loader and query CLI."""

from __future__ import annotations

import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

DEFAULT_URL = "postgresql://vehicle:vehicle@localhost:5432/vehicles"


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_URL)


def connect() -> psycopg.Connection:
    """Open a new connection using DATABASE_URL (or the local default)."""
    return psycopg.connect(database_url())
