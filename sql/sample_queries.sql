-- Which strategies are profitable?
SELECT
    strategy_name,
    COUNT(*) AS trades,
    ROUND(SUM(pnl_usd), 2) AS pnl_usd,
    ROUND(AVG(pnl_usd), 2) AS avg_trade_pnl
FROM trade_pnl
GROUP BY strategy_name
ORDER BY pnl_usd DESC;

-- Which strategy creates the most drawdown? Use the Python export for exact drawdown.
-- This SQL approximation shows the worst daily loss by strategy.
SELECT
    strategy_name,
    close_date,
    ROUND(SUM(pnl_usd), 2) AS daily_pnl_usd
FROM trade_pnl
GROUP BY strategy_name, close_date
ORDER BY daily_pnl_usd ASC
LIMIT 10;

-- Current risk exposure by strategy.
SELECT
    s.strategy_name,
    p.snapshot_date,
    ROUND(SUM(p.capital_used_usd), 2) AS capital_used_usd,
    ROUND(SUM(p.delta_exposure), 2) AS delta_exposure,
    ROUND(SUM(p.vega_exposure), 2) AS vega_exposure,
    ROUND(SUM(p.theta_exposure), 2) AS theta_exposure,
    ROUND(SUM(p.var_95_usd), 2) AS var_95_usd
FROM position_snapshots p
JOIN strategies s
    ON s.strategy_id = p.strategy_id
WHERE p.snapshot_date = (SELECT MAX(snapshot_date) FROM position_snapshots)
GROUP BY s.strategy_name, p.snapshot_date
ORDER BY var_95_usd DESC;

-- Is P&L concentrated in one lucky trade?
WITH ranked AS (
    SELECT
        strategy_name,
        trade_id,
        pnl_usd,
        SUM(pnl_usd) OVER (PARTITION BY strategy_name) AS strategy_pnl,
        ROW_NUMBER() OVER (PARTITION BY strategy_name ORDER BY pnl_usd DESC) AS pnl_rank
    FROM trade_pnl
)
SELECT
    strategy_name,
    trade_id AS largest_winner_trade_id,
    ROUND(pnl_usd, 2) AS largest_winner_usd,
    ROUND(strategy_pnl, 2) AS strategy_pnl_usd,
    ROUND(pnl_usd / NULLIF(strategy_pnl, 0), 4) AS largest_winner_contribution
FROM ranked
WHERE pnl_rank = 1
ORDER BY largest_winner_contribution DESC;

