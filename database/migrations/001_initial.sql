-- Tapewright-Quant Migration 001: Initial Relational Schema
-- Tables: assets, daily_market_summary, model_registry, trade_signals, order_executions, portfolio_snapshots

CREATE TABLE IF NOT EXISTS assets (
    asset_id VARCHAR(32) PRIMARY KEY,
    symbol VARCHAR(16) NOT NULL UNIQUE,
    name VARCHAR(128) NOT NULL,
    asset_class ENUM('EQUITY', 'CRYPTO', 'FX', 'COMMODITY', 'INDEX') NOT NULL,
    exchange VARCHAR(32) NOT NULL,
    currency VARCHAR(8) NOT NULL DEFAULT 'USD',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS daily_market_summary (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    asset_id VARCHAR(32) NOT NULL,
    trade_date DATE NOT NULL,
    open_price DECIMAL(18, 6) NOT NULL,
    high_price DECIMAL(18, 6) NOT NULL,
    low_price DECIMAL(18, 6) NOT NULL,
    close_price DECIMAL(18, 6) NOT NULL,
    volume DECIMAL(24, 6) NOT NULL,
    vwap DECIMAL(18, 6),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id) ON DELETE CASCADE,
    UNIQUE KEY uq_asset_date (asset_id, trade_date)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS model_registry (
    model_id VARCHAR(64) PRIMARY KEY,
    model_name VARCHAR(128) NOT NULL,
    model_family ENUM('TIME_SERIES', 'DEEP_LEARNING', 'ENSEMBLE', 'RL') NOT NULL,
    version VARCHAR(32) NOT NULL,
    hyperparameters JSON,
    metrics JSON,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS trade_signals (
    signal_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    asset_id VARCHAR(32) NOT NULL,
    model_id VARCHAR(64) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    signal_type ENUM('BUY', 'SELL', 'HOLD', 'REBALANCE') NOT NULL,
    confidence_score DECIMAL(5, 4) NOT NULL,
    target_weight DECIMAL(6, 4),
    metadata JSON,
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id),
    FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS order_executions (
    order_id VARCHAR(64) PRIMARY KEY,
    asset_id VARCHAR(32) NOT NULL,
    signal_id BIGINT,
    side ENUM('BUY', 'SELL') NOT NULL,
    order_type ENUM('MARKET', 'LIMIT', 'STOP') NOT NULL,
    quantity DECIMAL(18, 6) NOT NULL,
    executed_price DECIMAL(18, 6) NOT NULL,
    slippage_cost DECIMAL(18, 6) DEFAULT 0.0,
    fee DECIMAL(18, 6) DEFAULT 0.0,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id),
    FOREIGN KEY (signal_id) REFERENCES trade_signals(signal_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    snapshot_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    portfolio_name VARCHAR(64) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_nav DECIMAL(20, 4) NOT NULL,
    cash_balance DECIMAL(20, 4) NOT NULL,
    unrealized_pnl DECIMAL(20, 4) NOT NULL,
    realized_pnl DECIMAL(20, 4) NOT NULL,
    sharpe_ratio DECIMAL(8, 4),
    max_drawdown DECIMAL(6, 4)
) ENGINE=InnoDB;
