# Project Plan

## Step 1: Analytics Foundation

Status: Complete

- SQLite schema for strategies, trades, and current position snapshots
- Synthetic trading book with multiple strategies and realistic risk behavior
- Python calculations for P&L, drawdown, streaks, Sharpe, Sortino, VaR, Greeks, and P&L concentration
- Dashboard-ready exports
- Tests for key analytics helpers

## Step 2: Python Dashboard Build

Status: Complete

- Streamlit dashboard in `dashboard/app.py`
- Interactive filters for strategy and close-date range
- Overview, strategy performance, current risk, and trade quality tabs
- Plotly charts for P&L, drawdown, strategy P&L, VaR, Greeks, and trade distribution
- Current risk cards for delta, vega, theta, VaR, and capital utilization

## Step 3: Real Trade Journal Ingestion

Status: Planned

- Use `excel/trade_journal_template.csv` as the source format
- Add an ingestion script that validates required fields
- Load journal rows into SQLite
- Re-run the pipeline and refresh the Python dashboard

## Step 4: Advanced Risk Analytics

Status: Planned

- Add rolling Sharpe and rolling drawdown
- Add strategy correlation and P&L attribution
- Add scenario shocks for delta, vega, and rates exposure
- Add VaR by strategy and component VaR

## Step 5: Portfolio Presentation Polish

Status: Planned

- Add screenshots of final dashboard pages
- Write a recruiter-facing case study summary
- Add interview talking points for trading analytics and risk analytics roles
