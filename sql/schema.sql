PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS strategies (
    strategy_id INTEGER PRIMARY KEY,
    strategy_name TEXT NOT NULL UNIQUE,
    strategy_family TEXT NOT NULL,
    desk TEXT NOT NULL,
    risk_budget_usd REAL NOT NULL CHECK (risk_budget_usd >= 0),
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1))
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    strategy_id INTEGER NOT NULL,
    trade_date TEXT NOT NULL,
    close_date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    quantity REAL NOT NULL CHECK (quantity > 0),
    entry_price REAL NOT NULL CHECK (entry_price > 0),
    exit_price REAL NOT NULL CHECK (exit_price > 0),
    point_value REAL NOT NULL DEFAULT 1 CHECK (point_value > 0),
    fees REAL NOT NULL DEFAULT 0 CHECK (fees >= 0),
    risk_usd REAL NOT NULL CHECK (risk_usd >= 0),
    capital_used_usd REAL NOT NULL CHECK (capital_used_usd >= 0),
    delta_exposure REAL NOT NULL DEFAULT 0,
    gamma_exposure REAL NOT NULL DEFAULT 0,
    vega_exposure REAL NOT NULL DEFAULT 0,
    theta_exposure REAL NOT NULL DEFAULT 0,
    implied_vol REAL,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
);

CREATE TABLE IF NOT EXISTS position_snapshots (
    snapshot_date TEXT NOT NULL,
    strategy_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    quantity REAL NOT NULL,
    market_value_usd REAL NOT NULL,
    capital_used_usd REAL NOT NULL,
    delta_exposure REAL NOT NULL DEFAULT 0,
    gamma_exposure REAL NOT NULL DEFAULT 0,
    vega_exposure REAL NOT NULL DEFAULT 0,
    theta_exposure REAL NOT NULL DEFAULT 0,
    var_95_usd REAL NOT NULL DEFAULT 0,
    notes TEXT,
    PRIMARY KEY (snapshot_date, strategy_id, symbol),
    FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
);

CREATE VIEW IF NOT EXISTS trade_pnl AS
SELECT
    t.trade_id,
    t.strategy_id,
    s.strategy_name,
    s.strategy_family,
    t.trade_date,
    t.close_date,
    t.symbol,
    t.asset_class,
    t.side,
    t.quantity,
    t.entry_price,
    t.exit_price,
    t.point_value,
    t.fees,
    CASE
        WHEN t.side = 'LONG'
            THEN (t.exit_price - t.entry_price) * t.quantity * t.point_value
        ELSE (t.entry_price - t.exit_price) * t.quantity * t.point_value
    END AS gross_pnl_usd,
    CASE
        WHEN t.side = 'LONG'
            THEN (t.exit_price - t.entry_price) * t.quantity * t.point_value - t.fees
        ELSE (t.entry_price - t.exit_price) * t.quantity * t.point_value - t.fees
    END AS pnl_usd,
    t.risk_usd,
    t.capital_used_usd,
    t.delta_exposure,
    t.gamma_exposure,
    t.vega_exposure,
    t.theta_exposure,
    t.implied_vol,
    t.notes
FROM trades t
JOIN strategies s
    ON s.strategy_id = t.strategy_id;

