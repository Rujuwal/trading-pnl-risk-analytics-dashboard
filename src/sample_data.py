from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SAMPLE_END_DATE, SAMPLE_START_DATE
from .database import initialize_database, write_dataframe


@dataclass(frozen=True)
class StrategyProfile:
    strategy_id: int
    win_rate: float
    avg_win_r: float
    avg_loss_r: float
    capital_low: float
    capital_high: float
    risk_low: float
    risk_high: float


STRATEGIES = pd.DataFrame(
    [
        (1, "Momentum Breakout", "Directional", "Equity Index", 1_400_000, 1),
        (2, "Mean Reversion", "Stat Arb", "Equities", 1_100_000, 1),
        (3, "Options Vol Carry", "Volatility", "Options", 1_700_000, 1),
        (4, "Event Driven", "Catalyst", "Equities", 1_250_000, 1),
        (5, "Macro Futures", "Macro", "Futures", 1_500_000, 1),
        (6, "Index Hedging", "Hedging", "Portfolio Overlay", 750_000, 1),
    ],
    columns=[
        "strategy_id",
        "strategy_name",
        "strategy_family",
        "desk",
        "risk_budget_usd",
        "is_active",
    ],
)

PROFILES = {
    1: StrategyProfile(1, 0.51, 1.55, 1.00, 150_000, 900_000, 0.004, 0.013),
    2: StrategyProfile(2, 0.58, 1.10, 0.85, 120_000, 650_000, 0.003, 0.009),
    3: StrategyProfile(3, 0.69, 0.65, 1.85, 200_000, 1_250_000, 0.004, 0.015),
    4: StrategyProfile(4, 0.44, 1.85, 1.05, 100_000, 800_000, 0.005, 0.018),
    5: StrategyProfile(5, 0.50, 1.35, 1.05, 180_000, 1_000_000, 0.004, 0.014),
    6: StrategyProfile(6, 0.42, 0.75, 0.70, 80_000, 450_000, 0.002, 0.007),
}

SYMBOLS = {
    "Equity": [
        ("AAPL", 1, 185),
        ("MSFT", 1, 415),
        ("NVDA", 1, 875),
        ("TSLA", 1, 185),
        ("SPY", 1, 525),
        ("QQQ", 1, 455),
    ],
    "Future": [
        ("ES1!", 50, 5300),
        ("NQ1!", 20, 18500),
        ("CL1!", 1000, 78),
        ("GC1!", 100, 2350),
        ("ZN1!", 1000, 108),
    ],
    "Option": [
        ("SPX 202606 C5400", 100, 52),
        ("SPX 202606 P5000", 100, 44),
        ("NVDA 202606 C950", 100, 68),
        ("QQQ 202606 P430", 100, 31),
    ],
    "FX": [
        ("EURUSD", 100_000, 1.08),
        ("USDJPY", 100_000, 156),
        ("GBPUSD", 100_000, 1.27),
    ],
}

STRATEGY_ASSET_CLASS = {
    1: ["Equity", "Future"],
    2: ["Equity"],
    3: ["Option"],
    4: ["Equity", "Option"],
    5: ["Future", "FX"],
    6: ["Future", "Option"],
}


def generate_strategies() -> pd.DataFrame:
    return STRATEGIES.copy()


def _draw_r_multiple(profile: StrategyProfile, rng: np.random.Generator) -> float:
    if rng.random() <= profile.win_rate:
        return max(0.05, rng.normal(profile.avg_win_r, 0.45))
    return -max(0.05, rng.normal(profile.avg_loss_r, 0.50))


def _choose_symbol(strategy_id: int, rng: np.random.Generator) -> tuple[str, str, float, float]:
    asset_class = rng.choice(STRATEGY_ASSET_CLASS[strategy_id])
    symbol, point_value, anchor_price = SYMBOLS[asset_class][
        int(rng.integers(0, len(SYMBOLS[asset_class])))
    ]
    entry_price = max(0.25, rng.normal(anchor_price, anchor_price * 0.035))
    return symbol, asset_class, float(point_value), float(entry_price)


def _greeks_for_trade(
    strategy_id: int,
    asset_class: str,
    side_multiplier: int,
    capital_used: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float | None]:
    delta = side_multiplier * capital_used * rng.uniform(0.15, 1.05)
    gamma = 0.0
    vega = 0.0
    theta = 0.0
    implied_vol = None

    if asset_class == "Option":
        implied_vol = float(rng.uniform(0.18, 0.48))
        delta = side_multiplier * capital_used * rng.uniform(0.05, 0.40)
        gamma = side_multiplier * rng.uniform(700, 6_500)
        if strategy_id == 3:
            vega = -rng.uniform(12_000, 70_000)
            theta = rng.uniform(1_500, 9_500)
        else:
            vega = rng.uniform(6_000, 35_000)
            theta = -rng.uniform(800, 5_500)

    return float(delta), float(gamma), float(vega), float(theta), implied_vol


def generate_trades(seed: int = 42, trade_count: int = 540) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    business_days = pd.bdate_range(SAMPLE_START_DATE, SAMPLE_END_DATE)
    strategy_ids = np.array(list(PROFILES.keys()))
    strategy_probabilities = np.array([0.21, 0.19, 0.20, 0.13, 0.17, 0.10])
    rows = []

    for idx in range(1, trade_count + 1):
        strategy_id = int(rng.choice(strategy_ids, p=strategy_probabilities))
        profile = PROFILES[strategy_id]
        trade_day_index = int(rng.integers(0, len(business_days) - 12))
        holding_days = int(rng.integers(0, 12))
        trade_date = business_days[trade_day_index]
        close_date = business_days[min(trade_day_index + holding_days, len(business_days) - 1)]
        symbol, asset_class, point_value, entry_price = _choose_symbol(strategy_id, rng)
        side = str(rng.choice(["LONG", "SHORT"], p=[0.58, 0.42]))
        side_multiplier = 1 if side == "LONG" else -1

        capital_used = float(rng.uniform(profile.capital_low, profile.capital_high))
        risk_usd = float(capital_used * rng.uniform(profile.risk_low, profile.risk_high))
        r_multiple = _draw_r_multiple(profile, rng)

        if strategy_id == 3 and close_date >= pd.Timestamp("2026-02-01") and rng.random() < 0.11:
            r_multiple = -rng.uniform(3.0, 5.5)

        net_pnl = float(risk_usd * r_multiple)
        fees = float(max(2.0, capital_used * rng.uniform(0.000015, 0.00011)))
        quantity = max(1.0, round(capital_used / (entry_price * point_value), 0))
        gross_pnl = net_pnl + fees
        exit_price = entry_price + gross_pnl / (quantity * point_value * side_multiplier)
        exit_price = max(0.10, float(exit_price))

        delta, gamma, vega, theta, implied_vol = _greeks_for_trade(
            strategy_id,
            asset_class,
            side_multiplier,
            capital_used,
            rng,
        )

        rows.append(
            {
                "trade_id": f"T{idx:05d}",
                "strategy_id": strategy_id,
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "close_date": close_date.strftime("%Y-%m-%d"),
                "symbol": symbol,
                "asset_class": asset_class,
                "side": side,
                "quantity": quantity,
                "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price, 4),
                "point_value": point_value,
                "fees": round(fees, 2),
                "risk_usd": round(risk_usd, 2),
                "capital_used_usd": round(capital_used, 2),
                "delta_exposure": round(delta, 2),
                "gamma_exposure": round(gamma, 2),
                "vega_exposure": round(vega, 2),
                "theta_exposure": round(theta, 2),
                "implied_vol": None if implied_vol is None else round(implied_vol, 4),
                "notes": "Synthetic sample trade",
            }
        )

    rows.append(_lucky_event_trade())
    return pd.DataFrame(rows)


def _lucky_event_trade() -> dict:
    entry_price = 92.0
    capital_used = 1_250_000.0
    point_value = 1.0
    quantity = round(capital_used / (entry_price * point_value), 0)
    fees = 325.0
    net_pnl = 875_000.0
    exit_price = entry_price + (net_pnl + fees) / (quantity * point_value)

    return {
        "trade_id": "T-LUCKY-001",
        "strategy_id": 4,
        "trade_date": "2025-08-28",
        "close_date": "2025-09-15",
        "symbol": "NVDA",
        "asset_class": "Equity",
        "side": "LONG",
        "quantity": quantity,
        "entry_price": entry_price,
        "exit_price": round(exit_price, 4),
        "point_value": point_value,
        "fees": fees,
        "risk_usd": 150_000.0,
        "capital_used_usd": capital_used,
        "delta_exposure": 1_125_000.0,
        "gamma_exposure": 0.0,
        "vega_exposure": 0.0,
        "theta_exposure": 0.0,
        "implied_vol": None,
        "notes": "Injected outsized winner to test P&L concentration analysis",
    }


def generate_position_snapshots() -> pd.DataFrame:
    rows = [
        ("2026-06-04", 1, "NQ1!", "Future", 3, 1_125_000, 1_110_000, 1_050_000, 0, 0, 0, 82_000, "Long index momentum exposure"),
        ("2026-06-04", 1, "QQQ", "Equity", 1_800, 835_000, 820_000, 760_000, 0, 0, 0, 48_000, "Equity beta extension"),
        ("2026-06-04", 2, "MSFT", "Equity", 1_200, 505_000, 498_000, 265_000, 0, 0, 0, 22_000, "Pairs long leg"),
        ("2026-06-04", 2, "AAPL", "Equity", -1_450, -285_000, 290_000, -190_000, 0, 0, 0, 19_000, "Pairs short leg"),
        ("2026-06-04", 3, "SPX 202606 C5400", "Option", -90, -475_000, 620_000, -110_000, -8_500, -115_000, 11_800, 96_000, "Short call spread inventory"),
        ("2026-06-04", 3, "SPX 202606 P5000", "Option", -115, -510_000, 760_000, 85_000, -10_700, -162_000, 14_900, 124_000, "Short put spread inventory"),
        ("2026-06-04", 4, "TSLA", "Equity", 1_100, 214_000, 206_000, 180_000, 0, 0, 0, 31_000, "Catalyst position"),
        ("2026-06-04", 4, "NVDA 202606 C950", "Option", 30, 204_000, 220_000, 71_000, 2_300, 28_000, -3_200, 36_000, "Long upside optionality"),
        ("2026-06-04", 5, "CL1!", "Future", 4, 312_000, 305_000, 245_000, 0, 0, 0, 41_000, "Crude macro long"),
        ("2026-06-04", 5, "EURUSD", "FX", -3, -324_000, 330_000, -210_000, 0, 0, 0, 37_000, "USD long expression"),
        ("2026-06-04", 6, "ES1!", "Future", -2, -530_000, 535_000, -520_000, 0, 0, 0, 35_000, "Portfolio hedge"),
        ("2026-06-04", 6, "QQQ 202606 P430", "Option", 55, 171_000, 185_000, -125_000, 5_800, 62_000, -6_400, 42_000, "Crash convexity"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "snapshot_date",
            "strategy_id",
            "symbol",
            "asset_class",
            "quantity",
            "market_value_usd",
            "capital_used_usd",
            "delta_exposure",
            "gamma_exposure",
            "vega_exposure",
            "theta_exposure",
            "var_95_usd",
            "notes",
        ],
    )


def seed_database(reset: bool = True) -> None:
    initialize_database(reset=reset)
    write_dataframe("strategies", generate_strategies())
    write_dataframe("trades", generate_trades())
    write_dataframe("position_snapshots", generate_position_snapshots())

