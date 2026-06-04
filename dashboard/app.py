from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics import (  # noqa: E402
    annualized_sharpe,
    annualized_sortino,
    build_output_tables,
    expected_shortfall,
    export_output_tables,
    historical_var,
    max_consecutive_loss_streak,
    profit_factor_excluding_largest,
    profit_factor_excluding_none,
    write_summary_markdown,
)
from src.config import DASHBOARD_OUTPUT_DIR, PORTFOLIO_CAPITAL_USD  # noqa: E402
from src.database import read_table, table_row_count  # noqa: E402
from src.sample_data import seed_database  # noqa: E402

st.set_page_config(
    page_title="Trading P&L and Risk Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

GREEN = "#087f5b"
RED = "#c92a2a"
AMBER = "#b7791f"
CYAN = "#0b7285"
INK = "#1f2937"
MUTED = "#6b7280"
GRID = "#e5e7eb"


def apply_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }
        section[data-testid="stSidebar"] {
            border-right: 1px solid #e5e7eb;
        }
        h1, h2, h3 {
            color: #111827;
            letter-spacing: 0;
        }
        .desk-header {
            border-bottom: 1px solid #e5e7eb;
            padding-bottom: 0.75rem;
            margin-bottom: 1rem;
        }
        .desk-subtitle {
            color: #6b7280;
            font-size: 0.95rem;
            margin-top: -0.35rem;
        }
        .metric-card {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.85rem 0.9rem;
            min-height: 104px;
            background: #ffffff;
        }
        .metric-label {
            color: #6b7280;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            color: #111827;
            font-size: 1.45rem;
            line-height: 1.15;
            font-weight: 700;
            overflow-wrap: anywhere;
        }
        .metric-note {
            color: #6b7280;
            font-size: 0.78rem;
            margin-top: 0.35rem;
        }
        .good { color: #087f5b; }
        .bad { color: #c92a2a; }
        .warn { color: #b7791f; }
        .neutral { color: #1f2937; }
        div[data-testid="stDataFrame"] {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_dashboard_exports(force_refresh: bool = False) -> None:
    required = DASHBOARD_OUTPUT_DIR / "fact_trades.csv"
    if force_refresh or not required.exists() or table_row_count("trades") == 0:
        seed_database(reset=True)
        strategies = read_table("strategies")
        trades = read_table("trades")
        position_snapshots = read_table("position_snapshots")
        tables = build_output_tables(strategies, trades, position_snapshots)
        export_output_tables(tables)
        write_summary_markdown(tables)


@st.cache_data(ttl=60)
def load_tables() -> dict[str, pd.DataFrame]:
    ensure_dashboard_exports(force_refresh=False)
    tables = {}
    for path in DASHBOARD_OUTPUT_DIR.glob("*.csv"):
        tables[path.stem] = pd.read_csv(path)

    date_columns = {
        "fact_trades": ["trade_date", "close_date"],
        "fact_daily_pnl": ["date"],
        "fact_monthly_pnl": [],
        "fact_strategy_daily_pnl": ["date"],
        "fact_risk_exposure_today": ["snapshot_date"],
        "fact_risk_exposure_by_strategy": ["snapshot_date"],
        "portfolio_metrics": ["as_of_date"],
    }
    for table_name, columns in date_columns.items():
        if table_name in tables:
            for column in columns:
                if column in tables[table_name].columns:
                    tables[table_name][column] = pd.to_datetime(tables[table_name][column])
    return tables


def money(value: float, compact: bool = False) -> str:
    if pd.isna(value):
        return "n/a"
    sign = "-" if value < 0 else ""
    value = abs(float(value))
    if compact and value >= 1_000_000:
        return f"{sign}${value / 1_000_000:.2f}M"
    if compact and value >= 1_000:
        return f"{sign}${value / 1_000:.0f}K"
    return f"{sign}${value:,.0f}"


def number(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):,.{decimals}f}"


def pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.1%}"


def metric_card(label: str, value: str, note: str = "", tone: str = "neutral") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {tone}">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_layout(title: str, y_title: str = "") -> dict:
    return {
        "title": {"text": title, "font": {"size": 16}},
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "font": {"color": INK},
        "margin": {"l": 52, "r": 24, "t": 54, "b": 42},
        "xaxis": {"gridcolor": GRID, "zerolinecolor": GRID},
        "yaxis": {"gridcolor": GRID, "zerolinecolor": GRID, "title": y_title},
        "height": 360,
    }


def filtered_daily_from_trades(trades: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    if trades.empty:
        dates = pd.bdate_range(start_date, end_date)
        return pd.DataFrame(
            {
                "date": dates,
                "pnl_usd": 0.0,
                "cumulative_pnl_usd": 0.0,
                "drawdown_usd": 0.0,
                "daily_return": 0.0,
            }
        )

    daily = trades.groupby("close_date").agg(pnl_usd=("pnl_usd", "sum")).sort_index()
    all_days = pd.bdate_range(start_date, end_date)
    daily = daily.reindex(all_days, fill_value=0.0)
    daily.index.name = "date"
    daily["cumulative_pnl_usd"] = daily["pnl_usd"].cumsum()
    daily["running_peak_pnl_usd"] = daily["cumulative_pnl_usd"].cummax()
    daily["drawdown_usd"] = daily["cumulative_pnl_usd"] - daily["running_peak_pnl_usd"]
    daily["daily_return"] = daily["pnl_usd"] / PORTFOLIO_CAPITAL_USD
    return daily.reset_index()


def filtered_metrics(trades: pd.DataFrame, daily: pd.DataFrame, risk: pd.DataFrame) -> dict[str, float | str]:
    winners = trades.loc[trades["pnl_usd"] > 0, "pnl_usd"]
    losers = trades.loc[trades["pnl_usd"] < 0, "pnl_usd"]
    total_pnl = float(trades["pnl_usd"].sum()) if not trades.empty else 0.0
    largest_winner = float(winners.max()) if not winners.empty else 0.0
    net_vega = float(risk["vega_exposure"].sum()) if not risk.empty else 0.0

    if total_pnl <= 0 or winners.empty:
        skill_signal = "No positive edge yet"
    elif largest_winner / total_pnl > 0.35 and profit_factor_excluding_largest(trades) < 1:
        skill_signal = "Highly dependent on one winner"
    elif largest_winner / total_pnl > 0.25:
        skill_signal = "Some concentration in top winner"
    else:
        skill_signal = "Distributed P&L profile"

    if net_vega < -100_000:
        vol_signal = "Short volatility risk"
    elif net_vega > 100_000:
        vol_signal = "Long volatility risk"
    else:
        vol_signal = "Vega near neutral"

    return {
        "total_pnl": total_pnl,
        "trade_count": int(len(trades)),
        "win_rate": float((trades["pnl_usd"] > 0).mean()) if not trades.empty else np.nan,
        "avg_winner": float(winners.mean()) if not winners.empty else 0.0,
        "avg_loser": float(losers.mean()) if not losers.empty else 0.0,
        "winner_loser_ratio": float(winners.mean() / abs(losers.mean())) if not winners.empty and not losers.empty else np.nan,
        "max_drawdown": float(daily["drawdown_usd"].min()) if not daily.empty else 0.0,
        "sharpe": annualized_sharpe(daily["daily_return"]),
        "sortino": annualized_sortino(daily["daily_return"]),
        "var_95": historical_var(daily["daily_return"], 0.95),
        "expected_shortfall_95": expected_shortfall(daily["daily_return"], 0.95),
        "loss_streak": max_consecutive_loss_streak(trades) if not trades.empty else 0,
        "avg_risk": float(trades["risk_usd"].mean()) if not trades.empty else 0.0,
        "capital_utilization": float(risk["capital_used_usd"].sum() / PORTFOLIO_CAPITAL_USD) if not risk.empty else 0.0,
        "delta": float(risk["delta_exposure"].sum()) if not risk.empty else 0.0,
        "vega": net_vega,
        "theta": float(risk["theta_exposure"].sum()) if not risk.empty else 0.0,
        "current_var": float(risk["var_95_usd"].sum()) if not risk.empty else 0.0,
        "largest_winner": largest_winner,
        "largest_winner_pct": largest_winner / total_pnl if total_pnl > 0 else np.nan,
        "profit_factor": profit_factor_excluding_none(trades) if not trades.empty else np.nan,
        "profit_factor_ex_largest": profit_factor_excluding_largest(trades) if not trades.empty else np.nan,
        "skill_signal": skill_signal,
        "vol_signal": vol_signal,
    }


def line_and_drawdown(daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["cumulative_pnl_usd"],
            mode="lines",
            line={"color": GREEN, "width": 2.5},
            name="Cumulative P&L",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["drawdown_usd"],
            mode="lines",
            line={"color": RED, "width": 1.8},
            fill="tozeroy",
            name="Drawdown",
            yaxis="y2",
        )
    )
    fig.update_layout(**plot_layout("Cumulative P&L and Drawdown", "P&L"))
    fig.update_layout(
        yaxis2={
            "title": "Drawdown",
            "overlaying": "y",
            "side": "right",
            "gridcolor": "rgba(0,0,0,0)",
            "zerolinecolor": GRID,
        },
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    return fig


def bar_chart(frame: pd.DataFrame, x: str, y: str, title: str, color_by_sign: bool = True) -> go.Figure:
    colors = [GREEN if value >= 0 else RED for value in frame[y]] if color_by_sign else CYAN
    fig = go.Figure(go.Bar(x=frame[x], y=frame[y], marker={"color": colors}))
    fig.update_layout(**plot_layout(title, y))
    return fig


def exposure_chart(risk: pd.DataFrame) -> go.Figure:
    rows = []
    for _, row in risk.iterrows():
        for exposure in ["delta_exposure", "vega_exposure", "theta_exposure"]:
            rows.append(
                {
                    "strategy_name": row["strategy_name"],
                    "exposure": exposure.replace("_exposure", "").title(),
                    "value": row[exposure],
                }
            )
    exposure = pd.DataFrame(rows)
    fig = go.Figure()
    for name, color in [("Delta", CYAN), ("Vega", AMBER), ("Theta", GREEN)]:
        part = exposure.loc[exposure["exposure"] == name]
        fig.add_trace(go.Bar(x=part["strategy_name"], y=part["value"], name=name, marker={"color": color}))
    fig.update_layout(**plot_layout("Greeks Exposure by Strategy", "Exposure"))
    fig.update_layout(barmode="group", legend={"orientation": "h", "y": 1.08, "x": 0})
    return fig


def render_overview(metrics: dict[str, float | str], daily: pd.DataFrame, monthly: pd.DataFrame) -> None:
    cols = st.columns(4)
    with cols[0]:
        metric_card("Total P&L", money(metrics["total_pnl"], compact=True), "Filtered realized P&L", "good" if metrics["total_pnl"] >= 0 else "bad")
    with cols[1]:
        metric_card("Sharpe", number(metrics["sharpe"]), "Annualized daily return", "neutral")
    with cols[2]:
        metric_card("Max Drawdown", money(metrics["max_drawdown"], compact=True), "Worst peak-to-trough move", "bad")
    with cols[3]:
        metric_card("VaR 95%", money(metrics["var_95"], compact=True), "Historical one-day VaR", "warn")

    cols = st.columns([2, 1])
    with cols[0]:
        st.plotly_chart(line_and_drawdown(daily), width="stretch")
    with cols[1]:
        monthly_view = monthly.copy().tail(14)
        st.plotly_chart(bar_chart(monthly_view, "month", "pnl_usd", "Monthly P&L"), width="stretch")

    cols = st.columns(4)
    with cols[0]:
        metric_card("Win Rate", pct(metrics["win_rate"]), f"{metrics['trade_count']} closed trades")
    with cols[1]:
        metric_card("Avg Winner / Loser", number(metrics["winner_loser_ratio"]), f"{money(metrics['avg_winner'])} vs {money(metrics['avg_loser'])}")
    with cols[2]:
        metric_card("Capital Utilization", pct(metrics["capital_utilization"]), "Current snapshot")
    with cols[3]:
        metric_card("Sortino", number(metrics["sortino"]), "Downside volatility adjusted")


def render_strategy(strategy_summary: pd.DataFrame) -> None:
    strategy_view = strategy_summary.sort_values("pnl_usd", ascending=False)
    cols = st.columns([1, 1])
    with cols[0]:
        st.plotly_chart(bar_chart(strategy_view, "strategy_name", "pnl_usd", "Strategy-wise P&L"), width="stretch")
    with cols[1]:
        scatter = go.Figure(
            go.Scatter(
                x=strategy_view["max_drawdown_usd"].abs(),
                y=strategy_view["pnl_usd"],
                mode="markers+text",
                text=strategy_view["strategy_name"],
                textposition="top center",
                marker={
                    "size": np.clip(strategy_view["trade_count"], 20, 120),
                    "color": strategy_view["win_rate"],
                    "colorscale": "Teal",
                    "showscale": True,
                    "colorbar": {"title": "Win rate"},
                    "line": {"width": 1, "color": "#ffffff"},
                },
            )
        )
        scatter.update_layout(**plot_layout("P&L vs Drawdown", "P&L"))
        scatter.update_xaxes(title="Absolute Max Drawdown")
        st.plotly_chart(scatter, width="stretch")

    table = strategy_view[
        [
            "strategy_name",
            "pnl_usd",
            "trade_count",
            "win_rate",
            "avg_winner_usd",
            "avg_loser_usd",
            "profit_factor",
            "max_drawdown_usd",
            "max_consecutive_loss_streak",
        ]
    ]
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "pnl_usd": st.column_config.NumberColumn("P&L", format="$%.0f"),
            "win_rate": st.column_config.NumberColumn("Win Rate", format="%.1%"),
            "avg_winner_usd": st.column_config.NumberColumn("Avg Winner", format="$%.0f"),
            "avg_loser_usd": st.column_config.NumberColumn("Avg Loser", format="$%.0f"),
            "profit_factor": st.column_config.NumberColumn("Profit Factor", format="%.2f"),
            "max_drawdown_usd": st.column_config.NumberColumn("Max Drawdown", format="$%.0f"),
        },
    )


def render_risk(metrics: dict[str, float | str], risk: pd.DataFrame, positions: pd.DataFrame) -> None:
    cols = st.columns(5)
    with cols[0]:
        metric_card("Delta", money(metrics["delta"], compact=True), "Directional exposure")
    with cols[1]:
        metric_card("Vega", money(metrics["vega"], compact=True), metrics["vol_signal"], "bad" if metrics["vega"] < -100_000 else "warn")
    with cols[2]:
        metric_card("Theta", money(metrics["theta"], compact=True), "Daily theta exposure")
    with cols[3]:
        metric_card("Current VaR 95%", money(metrics["current_var"], compact=True), "Open risk snapshot", "warn")
    with cols[4]:
        metric_card("Capital Used", pct(metrics["capital_utilization"]), money(metrics["capital_utilization"] * PORTFOLIO_CAPITAL_USD, compact=True))

    cols = st.columns([3, 2])
    with cols[0]:
        st.plotly_chart(exposure_chart(risk), width="stretch")
    with cols[1]:
        var_frame = risk.sort_values("var_95_usd", ascending=False)
        st.plotly_chart(bar_chart(var_frame, "strategy_name", "var_95_usd", "Current VaR by Strategy", False), width="stretch")

    st.dataframe(
        positions[
            [
                "snapshot_date",
                "strategy_name",
                "symbol",
                "asset_class",
                "quantity",
                "capital_used_usd",
                "delta_exposure",
                "vega_exposure",
                "theta_exposure",
                "var_95_usd",
            ]
        ].sort_values("var_95_usd", ascending=False),
        width="stretch",
        hide_index=True,
        column_config={
            "capital_used_usd": st.column_config.NumberColumn("Capital Used", format="$%.0f"),
            "delta_exposure": st.column_config.NumberColumn("Delta", format="$%.0f"),
            "vega_exposure": st.column_config.NumberColumn("Vega", format="$%.0f"),
            "theta_exposure": st.column_config.NumberColumn("Theta", format="$%.0f"),
            "var_95_usd": st.column_config.NumberColumn("VaR 95%", format="$%.0f"),
        },
    )


def render_trade_quality(metrics: dict[str, float | str], trades: pd.DataFrame) -> None:
    cols = st.columns(5)
    with cols[0]:
        metric_card("Profit Factor", number(metrics["profit_factor"]), "Gross profit / gross loss")
    with cols[1]:
        metric_card("PF Ex Largest", number(metrics["profit_factor_ex_largest"]), "Outlier-adjusted")
    with cols[2]:
        metric_card("Largest Winner", money(metrics["largest_winner"], compact=True), pct(metrics["largest_winner_pct"]))
    with cols[3]:
        metric_card("Loss Streak", str(metrics["loss_streak"]), "Max consecutive losses")
    with cols[4]:
        metric_card("Risk / Trade", money(metrics["avg_risk"], compact=True), metrics["skill_signal"], "warn")

    cols = st.columns([3, 2])
    with cols[0]:
        histogram = go.Figure(go.Histogram(x=trades["pnl_usd"], nbinsx=45, marker={"color": CYAN}))
        histogram.update_layout(**plot_layout("Trade P&L Distribution", "Trades"))
        histogram.update_xaxes(title="Trade P&L")
        st.plotly_chart(histogram, width="stretch")
    with cols[1]:
        r_by_strategy = trades.groupby("strategy_name").agg(avg_r_multiple=("r_multiple", "mean")).reset_index()
        st.plotly_chart(bar_chart(r_by_strategy, "strategy_name", "avg_r_multiple", "Average R Multiple"), width="stretch")

    top_trades = trades.sort_values("pnl_usd", ascending=False).head(15)
    st.dataframe(
        top_trades[
            [
                "trade_id",
                "strategy_name",
                "close_date",
                "symbol",
                "side",
                "pnl_usd",
                "risk_usd",
                "r_multiple",
                "capital_used_usd",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "pnl_usd": st.column_config.NumberColumn("P&L", format="$%.0f"),
            "risk_usd": st.column_config.NumberColumn("Risk", format="$%.0f"),
            "r_multiple": st.column_config.NumberColumn("R Multiple", format="%.2f"),
            "capital_used_usd": st.column_config.NumberColumn("Capital Used", format="$%.0f"),
        },
    )


def main() -> None:
    apply_css()
    st.markdown(
        """
        <div class="desk-header">
            <h1>Trading P&L and Risk Analytics</h1>
            <div class="desk-subtitle">Realized performance, drawdown, trade quality, Greeks, and current portfolio risk.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button("Regenerate sample book"):
        ensure_dashboard_exports(force_refresh=True)
        st.cache_data.clear()
        st.rerun()

    tables = load_tables()
    trades = tables["fact_trades"]
    monthly = tables["fact_monthly_pnl"]
    strategy_summary = tables["strategy_summary"]
    risk = tables["fact_risk_exposure_by_strategy"]
    positions = tables["fact_risk_exposure_today"]

    strategies = sorted(tables["dim_strategy"]["strategy_name"].unique().tolist())
    selected_strategies = st.sidebar.multiselect("Strategy", strategies, default=strategies)

    min_date = trades["close_date"].min().date()
    max_date = max(trades["close_date"].max(), positions["snapshot_date"].max()).date()
    selected_dates = st.sidebar.date_input(
        "Closed trade date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = [pd.Timestamp(value) for value in selected_dates]
    else:
        start_date = end_date = pd.Timestamp(selected_dates)

    filtered_trades = trades.loc[
        trades["strategy_name"].isin(selected_strategies)
        & (trades["close_date"] >= start_date)
        & (trades["close_date"] <= end_date)
    ].copy()
    filtered_risk = risk.loc[risk["strategy_name"].isin(selected_strategies)].copy()
    filtered_positions = positions.loc[positions["strategy_name"].isin(selected_strategies)].copy()
    filtered_strategy_summary = strategy_summary.loc[strategy_summary["strategy_name"].isin(selected_strategies)].copy()
    daily = filtered_daily_from_trades(filtered_trades, start_date, end_date)
    metrics = filtered_metrics(filtered_trades, daily, filtered_risk)

    st.sidebar.caption(f"Database exports: `{DASHBOARD_OUTPUT_DIR}`")
    st.sidebar.caption(f"As of: `{max_date.isoformat()}`")

    overview, strategy, risk_tab, quality = st.tabs(
        ["Overview", "Strategy Performance", "Current Risk", "Trade Quality"]
    )
    with overview:
        render_overview(metrics, daily, monthly)
    with strategy:
        render_strategy(filtered_strategy_summary)
    with risk_tab:
        render_risk(metrics, filtered_risk, filtered_positions)
    with quality:
        render_trade_quality(metrics, filtered_trades)


if __name__ == "__main__":
    main()
