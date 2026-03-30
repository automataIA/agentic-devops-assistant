"""Synthetic log seeder — generates realistic SRE log data into DuckDB.

Writes log rows directly into an in-process DuckDB table (and optionally exports
to Parquet for Iceberg ingestion).

Usage:
    uv run -m src.tools.seed_logs --rows 10000
    uv run -m src.tools.seed_logs --rows 5000 --output ./data/logs.parquet
"""

from __future__ import annotations

import argparse
import asyncio
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb

# ── Seed data ─────────────────────────────────────────────────────────────────

SERVICES = [
    "api-gateway",
    "auth-service",
    "payment-service",
    "notification-service",
    "user-service",
    "order-service",
    "inventory-service",
    "search-service",
]

LEVELS = ["DEBUG", "INFO", "INFO", "INFO", "WARN", "ERROR", "CRITICAL"]

ERROR_MESSAGES: list[tuple[str, str]] = [
    ("api-gateway",        "Upstream timeout after 30s"),
    ("api-gateway",        "Circuit breaker OPEN for auth-service"),
    ("auth-service",       "JWT signature verification failed"),
    ("auth-service",       "Redis connection refused"),
    ("payment-service",    "Stripe API returned 502"),
    ("payment-service",    "Transaction rollback: deadlock detected"),
    ("notification-service","SMTP connection timeout"),
    ("user-service",       "Database pool exhausted (max=50)"),
    ("order-service",      "OOMKilled: container exceeded memory limit"),
    ("inventory-service",  "Kafka consumer lag exceeded threshold: 50000"),
]

INFO_MESSAGES = [
    "Request processed successfully",
    "Cache hit ratio: 94.2%",
    "Health check passed",
    "Deployment completed",
    "Metrics pushed to Prometheus",
    "Config reloaded",
    "Connection pool warmed up",
    "Graceful shutdown initiated",
]

WARN_MESSAGES = [
    "High memory usage: 85%",
    "Slow query detected (>500ms)",
    "Certificate expires in 14 days",
    "Retry attempt 2/3",
    "Rate limit approaching: 90% of quota used",
]


def _generate_rows(n: int) -> list[dict]:
    rows = []
    now = datetime.now(tz=UTC)

    for i in range(n):
        # Distribute timestamps over the last 7 days with higher density in last 24h
        if random.random() < 0.7:
            ts = now - timedelta(hours=random.uniform(0, 24))
        else:
            ts = now - timedelta(days=random.uniform(1, 7))

        service = random.choice(SERVICES)
        level = random.choice(LEVELS)

        if level in ("ERROR", "CRITICAL"):
            # Pick a realistic error for the service if available
            matching = [m for s, m in ERROR_MESSAGES if s == service]
            message = random.choice(matching) if matching else random.choice([m for _, m in ERROR_MESSAGES])
        elif level == "WARN":
            message = random.choice(WARN_MESSAGES)
        else:
            message = random.choice(INFO_MESSAGES)

        trace_id = f"{random.randint(0, 0xFFFFFFFF):08x}{random.randint(0, 0xFFFFFFFF):08x}"

        rows.append({
            "timestamp": ts.isoformat(),
            "service": service,
            "level": level,
            "message": message,
            "trace_id": trace_id,
        })

    return rows


async def seed_logs(rows: int = 10_000, output_path: str | None = None) -> None:
    """Generate synthetic log rows and insert them into DuckDB.

    Args:
        rows: Number of log rows to generate.
        output_path: Optional path to export as Parquet (e.g., ./data/logs.parquet).
    """
    print(f"Generating {rows:,} synthetic log rows …")
    data = _generate_rows(rows)

    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE logs (
            timestamp VARCHAR,
            service   VARCHAR,
            level     VARCHAR,
            message   VARCHAR,
            trace_id  VARCHAR
        )
    """)

    conn.executemany(
        "INSERT INTO logs VALUES (?, ?, ?, ?, ?)",
        [(r["timestamp"], r["service"], r["level"], r["message"], r["trace_id"]) for r in data],
    )

    count = conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
    print(f"Inserted {count:,} rows.")

    # Level breakdown
    breakdown = conn.execute(
        "SELECT level, COUNT(*) as cnt FROM logs GROUP BY level ORDER BY cnt DESC"
    ).fetchdf()
    print("\nLevel distribution:")
    print(breakdown.to_string(index=False))

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        conn.execute(f"COPY logs TO '{out}' (FORMAT PARQUET)")
        print(f"\nExported to: {out}")

    conn.close()
    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed synthetic log data into DuckDB.")
    parser.add_argument("--rows", type=int, default=10_000, help="Number of rows to generate")
    parser.add_argument("--output", help="Optional Parquet output path")
    args = parser.parse_args()
    asyncio.run(seed_logs(rows=args.rows, output_path=args.output))


if __name__ == "__main__":
    main()
