from __future__ import annotations

import argparse

from .analytics import build_output_tables, export_output_tables, write_summary_markdown
from .config import DASHBOARD_OUTPUT_DIR, DB_PATH, SUMMARY_OUTPUT_DIR, ensure_directories
from .database import initialize_database, read_table, table_row_count
from .sample_data import seed_database


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build trading P&L and risk analytics outputs.")
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Recreate the SQLite database and seed it with sample trading data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    if args.reset_db or table_row_count("trades") == 0:
        seed_database(reset=True)
    else:
        initialize_database(reset=False)

    strategies = read_table("strategies")
    trades = read_table("trades")
    position_snapshots = read_table("position_snapshots")

    tables = build_output_tables(strategies, trades, position_snapshots)
    export_output_tables(tables)
    write_summary_markdown(tables)

    metrics = tables["portfolio_metrics"].iloc[0]
    print("Trading analytics pipeline complete.")
    print(f"Database: {DB_PATH}")
    print(f"Dashboard exports: {DASHBOARD_OUTPUT_DIR}")
    print(f"Summary: {SUMMARY_OUTPUT_DIR / 'latest_dashboard_summary.md'}")
    print(f"Total P&L: ${metrics['total_pnl_usd']:,.0f}")
    print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"Portfolio VaR 95%: ${metrics['portfolio_var_95_usd']:,.0f}")
    print(f"Skill vs luck: {metrics['skill_vs_luck_signal']}")


if __name__ == "__main__":
    main()
