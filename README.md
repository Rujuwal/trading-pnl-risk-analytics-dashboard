# Trading P&L and Risk Analytics Dashboard

This project builds a trading desk style analytics stack:

- SQLite trade database for raw trades and current risk snapshots
- Python analytics pipeline for P&L, drawdown, risk, ratios, Greeks, and VaR
- Streamlit dashboard for interactive trading analytics
- Optional Excel/CSV trade journal input template

## Business Questions Answered

- Which strategy is profitable?
- Which strategy creates the most drawdown?
- What is the risk exposure today?
- Is the trader overexposed to volatility?
- Is P&L coming from repeatable skill or one lucky trade?

## Project Structure

```text
.
|-- data/                         # SQLite database is generated here
|-- dashboard/                    # Python dashboard app
|-- excel/                        # Optional trade journal template
|-- outputs/dashboard/            # Dashboard-ready CSV exports
|-- sql/                          # Database schema and useful SQL questions
|-- src/                          # Python calculation pipeline
`-- tests/                        # Focused analytics tests
```

## Step 1: Create the Analytics Dataset

From this project folder:

```powershell
python -m pip install -r requirements.txt
python -m src.run_pipeline --reset-db
```

This creates:

- `data/trading_analytics.db`
- `outputs/dashboard/dim_strategy.csv`
- `outputs/dashboard/fact_trades.csv`
- `outputs/dashboard/fact_daily_pnl.csv`
- `outputs/dashboard/fact_monthly_pnl.csv`
- `outputs/dashboard/fact_strategy_daily_pnl.csv`
- `outputs/dashboard/fact_risk_exposure_today.csv`
- `outputs/dashboard/fact_risk_exposure_by_strategy.csv`
- `outputs/dashboard/portfolio_metrics.csv`
- `outputs/summary/latest_dashboard_summary.md`

## Step 2: Validate the Calculations

```powershell
python -m pytest
```

## Step 3: Run the Python Dashboard

```powershell
python -m streamlit run dashboard/app.py
```

The dashboard includes:

1. Overview
2. Strategy Performance
3. Current Risk
4. Trade Quality

## Step 4: Replace Sample Data With Real Trades

Use [excel/trade_journal_template.csv](excel/trade_journal_template.csv) as the input format. Then load those rows into the `trades` table in `data/trading_analytics.db`, or adapt `src/sample_data.py` into a real ingestion script.
