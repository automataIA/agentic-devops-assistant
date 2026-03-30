"""Seed Iceberg tables on MinIO using PyIceberg + SQLite catalog.

Reads data/logs_seed.parquet and writes it to s3://warehouse/iceberg/logs/app_logs/
via a SQLite-backed PyIceberg catalog.

Usage:
    uv run -m src.tools.seed_iceberg

Environment variables (or .env):
    ICEBERG_CATALOG_URI   — SQLite URI, e.g. sqlite:///./data/iceberg_catalog.db
    ICEBERG_WAREHOUSE     — S3 warehouse root, e.g. s3://warehouse/iceberg
    MINIO_ENDPOINT        — e.g. http://localhost:9000
    MINIO_ACCESS_KEY      — default: minioadmin
    MINIO_SECRET_KEY      — default: minioadmin
    DEMO_LOGS_PATH        — local parquet to seed, default: ./data/logs_seed.parquet
"""

from __future__ import annotations

import os
import sys

import pyarrow.parquet as pq
from dotenv import load_dotenv
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.exceptions import NamespaceAlreadyExistsError, NoSuchTableError

load_dotenv()


def _build_catalog() -> SqlCatalog:
    catalog_uri = os.getenv("ICEBERG_CATALOG_URI", "sqlite:///./data/iceberg_catalog.db")
    warehouse = os.getenv("ICEBERG_WAREHOUSE", "s3://warehouse/iceberg")
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")

    return SqlCatalog(
        "local",
        **{
            "uri": catalog_uri,
            "warehouse": warehouse,
            "s3.endpoint": minio_endpoint,
            "s3.access-key-id": access_key,
            "s3.secret-access-key": secret_key,
            "s3.region": "us-east-1",
            "s3.path-style-access": "true",
        },
    )


def seed_logs(catalog: SqlCatalog, parquet_path: str) -> None:
    """Write logs parquet file as an Iceberg table on MinIO."""
    print(f"[seed] Reading {parquet_path} …")
    arrow_table = pq.read_table(parquet_path)
    print(f"[seed] {len(arrow_table):,} rows, schema: {arrow_table.schema}")

    try:
        catalog.create_namespace("logs")
        print("[seed] Namespace 'logs' created.")
    except NamespaceAlreadyExistsError:
        print("[seed] Namespace 'logs' already exists — skipping.")

    # Drop existing table so we can re-seed cleanly
    try:
        catalog.drop_table("logs.app_logs")
        print("[seed] Dropped existing 'logs.app_logs' table.")
    except NoSuchTableError:
        pass

    iceberg_table = catalog.create_table(
        "logs.app_logs",
        schema=arrow_table.schema,
    )
    iceberg_table.append(arrow_table)

    print("[seed] Written to Iceberg table 'logs.app_logs'")
    print(f"[seed] Metadata location: {iceberg_table.metadata_location}")


def main() -> None:
    parquet_path = os.getenv("DEMO_LOGS_PATH", "./data/logs_seed.parquet")
    if not os.path.exists(parquet_path):
        print(f"[seed] ERROR: parquet file not found at {parquet_path}")
        print("[seed] Generate it first with: uv run -m src.tools.seed_logs --rows 5000")
        sys.exit(1)

    catalog = _build_catalog()
    seed_logs(catalog, parquet_path)
    print("[seed] Done.")


if __name__ == "__main__":
    main()
