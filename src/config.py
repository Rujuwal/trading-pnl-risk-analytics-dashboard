from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DASHBOARD_OUTPUT_DIR = OUTPUT_DIR / "dashboard"
SUMMARY_OUTPUT_DIR = OUTPUT_DIR / "summary"
SQL_DIR = PROJECT_ROOT / "sql"
DB_PATH = DATA_DIR / "trading_analytics.db"
SCHEMA_PATH = SQL_DIR / "schema.sql"

PORTFOLIO_CAPITAL_USD = 10_000_000
SAMPLE_START_DATE = "2024-01-02"
SAMPLE_END_DATE = "2026-06-04"


def ensure_directories() -> None:
    for path in [DATA_DIR, DASHBOARD_OUTPUT_DIR, SUMMARY_OUTPUT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
