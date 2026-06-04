import pandas as pd

from src.analytics import max_consecutive_loss_streak, max_drawdown_from_pnl, safe_divide


def test_max_drawdown_from_pnl_series():
    pnl = pd.Series([100, -40, -90, 30, -10, 140])
    assert max_drawdown_from_pnl(pnl) == -130


def test_max_consecutive_loss_streak_resets_on_winner():
    trades = pd.DataFrame(
        {
            "trade_id": ["T1", "T2", "T3", "T4", "T5", "T6"],
            "close_date": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06"]
            ),
            "loss_flag": [True, True, False, True, True, True],
        }
    )
    assert max_consecutive_loss_streak(trades) == 3


def test_safe_divide_handles_zero_denominator():
    result = safe_divide(pd.Series([10.0]), pd.Series([0.0]))
    assert pd.isna(result.iloc[0])

