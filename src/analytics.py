from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DASHBOARD_OUTPUT_DIR, PORTFOLIO_CAPITAL_USD, SUMMARY_OUTPUT_DIR

TRADING_DAYS_PER_YEAR = 252


def compute_trade_pnl(trades: pd.DataFrame, strategies: pd.DataFrame) -> pd.DataFrame:
    df = trades.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["close_date"] = pd.to_datetime(df["close_date"])
    side_multiplier = np.where(df["side"].str.upper() == "LONG", 1, -1)
    df["gross_pnl_usd"] = (
        (df["exit_price"] - df["entry_price"])
        * df["quantity"]
        * df["point_value"]
        * side_multiplier
    )
    df["pnl_usd"] = df["gross_pnl_usd"] - df["fees"]
    df["return_on_capital"] = safe_divide(df["pnl_usd"], df["capital_used_usd"])
    df["r_multiple"] = safe_divide(df["pnl_usd"], df["risk_usd"])
    df["holding_days"] = (df["close_date"] - df["trade_date"]).dt.days.clip(lower=0)
    df["win_flag"] = df["pnl_usd"] > 0
    df["loss_flag"] = df["pnl_usd"] < 0

    strategy_cols = ["strategy_id", "strategy_name", "strategy_family", "desk", "risk_budget_usd"]
    df = df.merge(strategies[strategy_cols], on="strategy_id", how="left")
    return df[
        [
            "trade_id",
            "strategy_id",
            "strategy_name",
            "strategy_family",
            "desk",
            "trade_date",
            "close_date",
            "symbol",
            "asset_class",
            "side",
            "quantity",
            "entry_price",
            "exit_price",
            "point_value",
            "fees",
            "gross_pnl_usd",
            "pnl_usd",
            "risk_usd",
            "capital_used_usd",
            "return_on_capital",
            "r_multiple",
            "holding_days",
            "win_flag",
            "loss_flag",
            "delta_exposure",
            "gamma_exposure",
            "vega_exposure",
            "theta_exposure",
            "implied_vol",
            "notes",
        ]
    ]


def safe_divide(numerator, denominator):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(numerator, denominator)
    if isinstance(result, pd.Series):
        return result.replace([np.inf, -np.inf], np.nan)
    if np.isinf(result):
        return np.nan
    return result


def max_drawdown_from_pnl(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    cumulative = pnl.cumsum()
    running_peak = cumulative.cummax()
    drawdown = cumulative - running_peak
    return float(drawdown.min())


def max_consecutive_loss_streak(trades: pd.DataFrame) -> int:
    if trades.empty:
        return 0

    ordered = trades.sort_values(["close_date", "trade_id"])
    max_streak = 0
    current_streak = 0
    for is_loss in ordered["loss_flag"]:
        if bool(is_loss):
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return int(max_streak)


def build_daily_pnl(trade_fact: pd.DataFrame, end_date: pd.Timestamp | None = None) -> pd.DataFrame:
    daily = (
        trade_fact.groupby("close_date")
        .agg(
            pnl_usd=("pnl_usd", "sum"),
            gross_pnl_usd=("gross_pnl_usd", "sum"),
            fees_usd=("fees", "sum"),
            trade_count=("trade_id", "count"),
            wins=("win_flag", "sum"),
            losses=("loss_flag", "sum"),
            capital_used_usd=("capital_used_usd", "sum"),
            risk_usd=("risk_usd", "sum"),
        )
        .sort_index()
    )

    max_close_date = trade_fact["close_date"].max()
    calendar_end_date = max_close_date if end_date is None else max(max_close_date, pd.Timestamp(end_date))
    all_business_days = pd.bdate_range(trade_fact["close_date"].min(), calendar_end_date)
    daily = daily.reindex(all_business_days, fill_value=0)
    daily.index.name = "date"
    daily["cumulative_pnl_usd"] = daily["pnl_usd"].cumsum()
    daily["equity_usd"] = PORTFOLIO_CAPITAL_USD + daily["cumulative_pnl_usd"]
    daily["daily_return"] = daily["pnl_usd"] / PORTFOLIO_CAPITAL_USD
    daily["running_peak_pnl_usd"] = daily["cumulative_pnl_usd"].cummax()
    daily["drawdown_usd"] = daily["cumulative_pnl_usd"] - daily["running_peak_pnl_usd"]
    daily["drawdown_pct"] = daily["drawdown_usd"] / PORTFOLIO_CAPITAL_USD
    daily["win_rate"] = safe_divide(daily["wins"], daily["trade_count"])
    return daily.reset_index()


def build_monthly_pnl(daily_pnl: pd.DataFrame) -> pd.DataFrame:
    monthly = daily_pnl.copy()
    monthly["month"] = pd.to_datetime(monthly["date"]).dt.to_period("M").astype(str)
    monthly = (
        monthly.groupby("month")
        .agg(
            pnl_usd=("pnl_usd", "sum"),
            trade_count=("trade_count", "sum"),
            wins=("wins", "sum"),
            losses=("losses", "sum"),
            fees_usd=("fees_usd", "sum"),
            risk_usd=("risk_usd", "sum"),
        )
        .reset_index()
    )
    monthly["monthly_return"] = monthly["pnl_usd"] / PORTFOLIO_CAPITAL_USD
    monthly["win_rate"] = safe_divide(monthly["wins"], monthly["trade_count"])
    return monthly


def build_strategy_daily_pnl(trade_fact: pd.DataFrame) -> pd.DataFrame:
    daily = (
        trade_fact.groupby(["strategy_id", "strategy_name", "strategy_family", "close_date"])
        .agg(
            pnl_usd=("pnl_usd", "sum"),
            trade_count=("trade_id", "count"),
            wins=("win_flag", "sum"),
            losses=("loss_flag", "sum"),
            risk_usd=("risk_usd", "sum"),
            capital_used_usd=("capital_used_usd", "sum"),
        )
        .reset_index()
        .rename(columns={"close_date": "date"})
        .sort_values(["strategy_id", "date"])
    )
    daily["cumulative_pnl_usd"] = daily.groupby("strategy_id")["pnl_usd"].cumsum()
    daily["running_peak_pnl_usd"] = daily.groupby("strategy_id")["cumulative_pnl_usd"].cummax()
    daily["drawdown_usd"] = daily["cumulative_pnl_usd"] - daily["running_peak_pnl_usd"]
    daily["win_rate"] = safe_divide(daily["wins"], daily["trade_count"])
    return daily


def build_strategy_summary(trade_fact: pd.DataFrame, strategy_daily: pd.DataFrame) -> pd.DataFrame:
    drawdowns = (
        strategy_daily.groupby("strategy_id")["drawdown_usd"]
        .min()
        .rename("max_drawdown_usd")
        .reset_index()
    )

    rows = []
    group_cols = ["strategy_id", "strategy_name", "strategy_family", "desk"]
    for keys, group in trade_fact.groupby(group_cols):
        strategy_id, strategy_name, strategy_family, desk = keys
        winners = group.loc[group["pnl_usd"] > 0, "pnl_usd"]
        losers = group.loc[group["pnl_usd"] < 0, "pnl_usd"]
        gross_profit = float(winners.sum())
        gross_loss = float(losers.sum())
        total_pnl = float(group["pnl_usd"].sum())
        largest_winner = float(winners.max()) if not winners.empty else 0.0
        pnl_ex_largest = total_pnl - largest_winner

        rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_name": strategy_name,
                "strategy_family": strategy_family,
                "desk": desk,
                "trade_count": int(group["trade_id"].count()),
                "pnl_usd": total_pnl,
                "gross_profit_usd": gross_profit,
                "gross_loss_usd": gross_loss,
                "avg_trade_pnl_usd": float(group["pnl_usd"].mean()),
                "median_trade_pnl_usd": float(group["pnl_usd"].median()),
                "wins": int(group["win_flag"].sum()),
                "losses": int(group["loss_flag"].sum()),
                "win_rate": safe_divide(group["win_flag"].sum(), group["trade_id"].count()),
                "win_loss_ratio": safe_divide(group["win_flag"].sum(), group["loss_flag"].sum()),
                "avg_winner_usd": float(winners.mean()) if not winners.empty else 0.0,
                "avg_loser_usd": float(losers.mean()) if not losers.empty else 0.0,
                "avg_winner_to_avg_loser": safe_divide(
                    winners.mean() if not winners.empty else np.nan,
                    abs(losers.mean()) if not losers.empty else np.nan,
                ),
                "profit_factor": safe_divide(gross_profit, abs(gross_loss)),
                "profit_factor_ex_largest_winner": profit_factor_excluding_largest(group),
                "largest_winner_usd": largest_winner,
                "largest_winner_pct_total_pnl": safe_divide(largest_winner, total_pnl),
                "pnl_ex_largest_winner_usd": pnl_ex_largest,
                "avg_risk_per_trade_usd": float(group["risk_usd"].mean()),
                "max_risk_per_trade_usd": float(group["risk_usd"].max()),
                "avg_r_multiple": float(group["r_multiple"].mean()),
                "max_consecutive_loss_streak": max_consecutive_loss_streak(group),
            }
        )

    summary = pd.DataFrame(rows).merge(drawdowns, on="strategy_id", how="left")
    summary["drawdown_to_pnl_ratio"] = safe_divide(
        summary["max_drawdown_usd"].abs(),
        summary["pnl_usd"].abs(),
    )
    return summary.sort_values("pnl_usd", ascending=False)


def profit_factor_excluding_largest(trades: pd.DataFrame) -> float:
    winners = trades.loc[trades["pnl_usd"] > 0].sort_values("pnl_usd", ascending=False)
    if winners.empty:
        return 0.0

    adjusted = trades.drop(index=winners.index[0])
    gross_profit = adjusted.loc[adjusted["pnl_usd"] > 0, "pnl_usd"].sum()
    gross_loss = adjusted.loc[adjusted["pnl_usd"] < 0, "pnl_usd"].sum()
    return safe_divide(float(gross_profit), abs(float(gross_loss)))


def build_risk_exposure(
    position_snapshots: pd.DataFrame,
    strategies: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = position_snapshots.copy()
    positions["snapshot_date"] = pd.to_datetime(positions["snapshot_date"])
    latest_date = positions["snapshot_date"].max()
    latest = positions.loc[positions["snapshot_date"] == latest_date].copy()
    latest = latest.merge(
        strategies[["strategy_id", "strategy_name", "strategy_family", "desk", "risk_budget_usd"]],
        on="strategy_id",
        how="left",
    )
    latest["capital_utilization_pct"] = latest["capital_used_usd"] / PORTFOLIO_CAPITAL_USD
    latest["var_95_pct_capital"] = latest["var_95_usd"] / PORTFOLIO_CAPITAL_USD
    latest["vega_abs_pct_capital"] = latest["vega_exposure"].abs() / PORTFOLIO_CAPITAL_USD

    by_strategy = (
        latest.groupby(["snapshot_date", "strategy_id", "strategy_name", "strategy_family", "desk"])
        .agg(
            market_value_usd=("market_value_usd", "sum"),
            capital_used_usd=("capital_used_usd", "sum"),
            delta_exposure=("delta_exposure", "sum"),
            gamma_exposure=("gamma_exposure", "sum"),
            vega_exposure=("vega_exposure", "sum"),
            theta_exposure=("theta_exposure", "sum"),
            var_95_usd=("var_95_usd", "sum"),
            risk_budget_usd=("risk_budget_usd", "first"),
        )
        .reset_index()
    )
    by_strategy["capital_utilization_pct"] = by_strategy["capital_used_usd"] / PORTFOLIO_CAPITAL_USD
    by_strategy["var_95_pct_capital"] = by_strategy["var_95_usd"] / PORTFOLIO_CAPITAL_USD
    by_strategy["risk_budget_used_pct"] = safe_divide(
        by_strategy["var_95_usd"],
        by_strategy["risk_budget_usd"],
    )
    by_strategy["volatility_posture"] = np.select(
        [
            by_strategy["vega_exposure"] < -100_000,
            by_strategy["vega_exposure"] > 100_000,
        ],
        ["Short volatility", "Long volatility"],
        default="Neutral volatility",
    )
    return latest, by_strategy


def annualized_sharpe(daily_returns: pd.Series) -> float:
    clean = daily_returns.dropna()
    if len(clean) < 2 or clean.std(ddof=1) == 0:
        return np.nan
    return float(math.sqrt(TRADING_DAYS_PER_YEAR) * clean.mean() / clean.std(ddof=1))


def annualized_sortino(daily_returns: pd.Series) -> float:
    clean = daily_returns.dropna()
    downside = clean.loc[clean < 0]
    if len(downside) < 2 or downside.std(ddof=1) == 0:
        return np.nan
    return float(math.sqrt(TRADING_DAYS_PER_YEAR) * clean.mean() / downside.std(ddof=1))


def historical_var(daily_returns: pd.Series, confidence: float) -> float:
    clean = daily_returns.dropna()
    if clean.empty:
        return 0.0
    percentile = (1 - confidence) * 100
    return float(max(0.0, -np.percentile(clean, percentile) * PORTFOLIO_CAPITAL_USD))


def expected_shortfall(daily_returns: pd.Series, confidence: float) -> float:
    clean = daily_returns.dropna()
    if clean.empty:
        return 0.0
    cutoff = np.percentile(clean, (1 - confidence) * 100)
    tail = clean.loc[clean <= cutoff]
    if tail.empty:
        return 0.0
    return float(max(0.0, -tail.mean() * PORTFOLIO_CAPITAL_USD))


def build_portfolio_metrics(
    trade_fact: pd.DataFrame,
    daily_pnl: pd.DataFrame,
    risk_by_strategy: pd.DataFrame,
) -> pd.DataFrame:
    daily = daily_pnl.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    risk_as_of_date = (
        pd.to_datetime(risk_by_strategy["snapshot_date"]).max()
        if not risk_by_strategy.empty
        else daily["date"].max()
    )
    as_of_date = max(daily["date"].max(), risk_as_of_date)
    current_month = daily["date"].dt.to_period("M") == as_of_date.to_period("M")
    current_year = daily["date"].dt.year == as_of_date.year
    winners = trade_fact.loc[trade_fact["pnl_usd"] > 0, "pnl_usd"]
    losers = trade_fact.loc[trade_fact["pnl_usd"] < 0, "pnl_usd"]
    largest_winner = float(winners.max()) if not winners.empty else 0.0
    total_pnl = float(trade_fact["pnl_usd"].sum())
    net_vega = float(risk_by_strategy["vega_exposure"].sum()) if not risk_by_strategy.empty else 0.0

    rows = [
        {
            "as_of_date": as_of_date.date().isoformat(),
            "portfolio_capital_usd": PORTFOLIO_CAPITAL_USD,
            "daily_pnl_usd": float(daily.loc[daily["date"] == as_of_date, "pnl_usd"].sum()),
            "mtd_pnl_usd": float(daily.loc[current_month, "pnl_usd"].sum()),
            "ytd_pnl_usd": float(daily.loc[current_year, "pnl_usd"].sum()),
            "total_pnl_usd": total_pnl,
            "total_return_pct": total_pnl / PORTFOLIO_CAPITAL_USD,
            "trade_count": int(trade_fact["trade_id"].count()),
            "win_rate": safe_divide(trade_fact["win_flag"].sum(), trade_fact["trade_id"].count()),
            "win_loss_ratio": safe_divide(trade_fact["win_flag"].sum(), trade_fact["loss_flag"].sum()),
            "average_winner_usd": float(winners.mean()) if not winners.empty else 0.0,
            "average_loser_usd": float(losers.mean()) if not losers.empty else 0.0,
            "average_winner_to_loser": safe_divide(
                winners.mean() if not winners.empty else np.nan,
                abs(losers.mean()) if not losers.empty else np.nan,
            ),
            "max_drawdown_usd": float(daily["drawdown_usd"].min()),
            "max_drawdown_pct": float(daily["drawdown_pct"].min()),
            "max_consecutive_loss_streak": max_consecutive_loss_streak(trade_fact),
            "avg_risk_per_trade_usd": float(trade_fact["risk_usd"].mean()),
            "max_risk_per_trade_usd": float(trade_fact["risk_usd"].max()),
            "capital_utilization_usd": float(risk_by_strategy["capital_used_usd"].sum()),
            "capital_utilization_pct": safe_divide(
                risk_by_strategy["capital_used_usd"].sum(),
                PORTFOLIO_CAPITAL_USD,
            ),
            "sharpe_ratio": annualized_sharpe(daily["daily_return"]),
            "sortino_ratio": annualized_sortino(daily["daily_return"]),
            "portfolio_var_95_usd": historical_var(daily["daily_return"], 0.95),
            "portfolio_var_99_usd": historical_var(daily["daily_return"], 0.99),
            "expected_shortfall_95_usd": expected_shortfall(daily["daily_return"], 0.95),
            "delta_exposure": float(risk_by_strategy["delta_exposure"].sum()),
            "gamma_exposure": float(risk_by_strategy["gamma_exposure"].sum()),
            "vega_exposure": net_vega,
            "theta_exposure": float(risk_by_strategy["theta_exposure"].sum()),
            "largest_winner_usd": largest_winner,
            "largest_winner_pct_total_pnl": safe_divide(largest_winner, total_pnl),
            "pnl_ex_largest_winner_usd": total_pnl - largest_winner,
            "profit_factor": profit_factor_excluding_none(trade_fact),
            "profit_factor_ex_largest_winner": profit_factor_excluding_largest(trade_fact),
            "skill_vs_luck_signal": skill_vs_luck_signal(trade_fact),
            "volatility_exposure_signal": volatility_signal(net_vega),
        }
    ]
    return pd.DataFrame(rows)


def profit_factor_excluding_none(trades: pd.DataFrame) -> float:
    gross_profit = trades.loc[trades["pnl_usd"] > 0, "pnl_usd"].sum()
    gross_loss = trades.loc[trades["pnl_usd"] < 0, "pnl_usd"].sum()
    return safe_divide(float(gross_profit), abs(float(gross_loss)))


def skill_vs_luck_signal(trades: pd.DataFrame) -> str:
    total_pnl = float(trades["pnl_usd"].sum())
    winners = trades.loc[trades["pnl_usd"] > 0, "pnl_usd"]
    if total_pnl <= 0 or winners.empty:
        return "No positive edge yet"

    largest_contribution = float(winners.max()) / total_pnl
    pf_ex_largest = profit_factor_excluding_largest(trades)
    if largest_contribution > 0.35 and pf_ex_largest < 1.0:
        return "Highly dependent on one winner"
    if largest_contribution > 0.25:
        return "Some concentration in top winner"
    return "Distributed P&L profile"


def volatility_signal(net_vega: float) -> str:
    if net_vega < -100_000:
        return "Short volatility risk"
    if net_vega > 100_000:
        return "Long volatility risk"
    return "Vega near neutral"


def build_output_tables(
    strategies: pd.DataFrame,
    trades: pd.DataFrame,
    position_snapshots: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    trade_fact = compute_trade_pnl(trades, strategies)
    latest_snapshot_date = pd.to_datetime(position_snapshots["snapshot_date"]).max()
    daily_pnl = build_daily_pnl(trade_fact, end_date=latest_snapshot_date)
    monthly_pnl = build_monthly_pnl(daily_pnl)
    strategy_daily = build_strategy_daily_pnl(trade_fact)
    strategy_summary = build_strategy_summary(trade_fact, strategy_daily)
    risk_today, risk_by_strategy = build_risk_exposure(position_snapshots, strategies)
    portfolio_metrics = build_portfolio_metrics(trade_fact, daily_pnl, risk_by_strategy)

    dim_strategy = strategies.copy()
    return {
        "dim_strategy": dim_strategy,
        "fact_trades": trade_fact,
        "fact_daily_pnl": daily_pnl,
        "fact_monthly_pnl": monthly_pnl,
        "fact_strategy_daily_pnl": strategy_daily,
        "strategy_summary": strategy_summary,
        "fact_risk_exposure_today": risk_today,
        "fact_risk_exposure_by_strategy": risk_by_strategy,
        "portfolio_metrics": portfolio_metrics,
    }


def export_output_tables(tables: dict[str, pd.DataFrame], output_dir: Path = DASHBOARD_OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        export_frame = frame.copy()
        for column in export_frame.select_dtypes(include=["datetime64[ns]"]).columns:
            export_frame[column] = export_frame[column].dt.strftime("%Y-%m-%d")
        export_frame.to_csv(output_dir / f"{name}.csv", index=False)


def write_summary_markdown(
    tables: dict[str, pd.DataFrame],
    output_path: Path = SUMMARY_OUTPUT_DIR / "latest_dashboard_summary.md",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = tables["portfolio_metrics"].iloc[0]
    strategy_summary = tables["strategy_summary"].copy()
    risk = tables["fact_risk_exposure_by_strategy"].copy()
    top_strategy = strategy_summary.iloc[0]
    worst_drawdown = strategy_summary.sort_values("max_drawdown_usd").iloc[0]
    largest_vega = risk.assign(abs_vega=lambda x: x["vega_exposure"].abs()).sort_values(
        "abs_vega",
        ascending=False,
    ).iloc[0]

    content = f"""# Latest Trading Dashboard Summary

As of: {metrics["as_of_date"]}

## Portfolio

- Total P&L: ${metrics["total_pnl_usd"]:,.0f}
- MTD P&L: ${metrics["mtd_pnl_usd"]:,.0f}
- YTD P&L: ${metrics["ytd_pnl_usd"]:,.0f}
- Sharpe ratio: {metrics["sharpe_ratio"]:.2f}
- Sortino ratio: {metrics["sortino_ratio"]:.2f}
- Max drawdown: ${metrics["max_drawdown_usd"]:,.0f}
- Portfolio VaR 95%: ${metrics["portfolio_var_95_usd"]:,.0f}
- Capital utilization: {metrics["capital_utilization_pct"]:.1%}

## Desk Questions

- Most profitable strategy: {top_strategy["strategy_name"]} (${top_strategy["pnl_usd"]:,.0f})
- Largest drawdown strategy: {worst_drawdown["strategy_name"]} (${worst_drawdown["max_drawdown_usd"]:,.0f})
- Largest vega exposure: {largest_vega["strategy_name"]} (${largest_vega["vega_exposure"]:,.0f})
- Volatility signal: {metrics["volatility_exposure_signal"]}
- Skill vs luck signal: {metrics["skill_vs_luck_signal"]}
- Largest winner contribution: {metrics["largest_winner_pct_total_pnl"]:.1%}
"""
    output_path.write_text(content, encoding="utf-8")
