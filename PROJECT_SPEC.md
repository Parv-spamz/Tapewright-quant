# PROJECT SPECIFICATION & ARCHITECTURAL HANDOVER: TAPEWRIGHT-QUANT ENGINE

> **Target Audience / Execution Agent**: Lead AI Quant Engineer  
> **Workspace Path**: `tapewright-quant/` (inside `larp projects` / `gradenest-tests`)  
> **Active Status**: Stage 1 Active Project  

---

## 1. Executive Summary & Project Mission

**Tapewright-Quant** is a modular, production-grade **Autonomous Quantitative Market Intelligence & Algorithmic Trading Framework**. It is designed specifically to serve as a centerpiece **Data Science & Quantitative Finance portfolio** demonstrating advanced applied mathematics, time-series econometrics, machine learning, deep learning, reinforcement learning, and event-driven systems simulation.

### Core Philosophy
1. **Algorithm & Math First**: The value of this project is concentrated in statistical rigor, feature engineering, mathematical modeling, and backtesting fidelity.
2. **Modular Granularity**: Every single directory represents a decoupled, specialized unit of logic. Every folder is an importable Python package (`__init__.py`).
3. **Iterative Extensibility**: The system is designed to start with clean foundational algorithms and scale incrementally toward live simulation and reinforcement learning.

---

## 2. Strict Technical & Language Constraints (NON-NEGOTIABLE)

When implementing, generating, or refactoring code for this project, you must strictly abide by the following constraints:

```text
======================================================================
1. ALLOWED LANGUAGES:      Python (3.10+) AND MySQL ONLY.
2. DISALLOWED TECHNOLOGIES: NO JavaScript, TypeScript, Node.js, Rust, C++.
3. DISALLOWED DATABASES:    NO Supabase, Firebase, MongoDB, PostgreSQL, 
                           DynamoDB, or third-party cloud DB services.
4. DATABASE SCOPE:         MINIMAL-DB PHILOSOPHY.
======================================================================
```

### The Minimal-DB Rule
- **What MySQL is used for**:
  - Relational metadata registries (`assets`, `model_registry`).
  - Quantitative signal audit trails (`trade_signals`).
  - Order execution records (`order_executions`).
  - High-level portfolio snapshots (`portfolio_snapshots`).
- **What MySQL is NOT used for**:
  - Do **NOT** store high-frequency tick data, gigabytes of OHLCV bars, or high-dimensional feature arrays in MySQL tables.
  - Large-scale historical price arrays, order book snapshots, and computed features must be handled in memory via **Pandas / NumPy / PyArrow / Parquet file storage** in `data/caching/parquet_store/`.

---

## 3. Recommended Python Scientific & Quantitative Stack

Only standard Python scientific libraries should be utilized:
- **Numerical & Tabular**: `numpy`, `scipy`, `pandas`, `pyarrow`
- **Statistical & Econometrics**: `statsmodels`, `arch` (for GARCH modeling), `scikit-learn`
- **Deep Learning**: `torch` (PyTorch for Temporal Fusion Transformers, LSTM, WaveNet)
- **Gradient Boosting**: `xgboost`, `lightgbm`, `catboost`
- **Reinforcement Learning**: `gymnasium` (custom trading environment)
- **Database Connector**: `mysql-connector-python` or `PyMySQL` (with raw parameterized SQL or lightweight `SQLAlchemy` Core)
- **Testing & Verification**: `pytest`, `unittest`

---

## 4. End-to-End System Data Flow

```text
[Market / News Feeds] (Yahoo, CCXT, FRED, EDGAR)
        │
        ▼
[Data Ingestion & Cleaning] ──► [Parquet Local Cache]
        │
        ▼
[Feature Engineering Engine]
   ├── Technical Indicators (Momentum, Trend, Volatility, Volume)
   ├── Market Microstructure (Order Book Imbalance, Tick Entropy)
   ├── Econometrics (Cointegration, Fractional Differentiation, Kalman Filter)
   └── Sentiment Signals (Lexicon Scoring, FinBERT Embeddings)
        │
        ▼
[Predictive & Generative Models]
   ├── Time Series (ARIMA, GARCH Volatility, Prophet)
   ├── Ensembles & DL (XGBoost, LSTMs, WaveNet, TFT)
   └── RL Agents (PPO / DQN Order Execution)
        │
        ▼
[Signal Generation & Arbitrage Filter] ──► Log to MySQL (`trade_signals`)
        │
        ▼
[Event-Driven Backtesting Engine]
   ├── Slippage & Transaction Cost Models
   ├── Portfolio Optimization (Markowitz, Black-Litterman, Risk Parity)
   └── Order Execution Simulation ──► Log to MySQL (`order_executions`)
        │
        ▼
[Performance Metrics & Tearsheet Analytics] ──► Log to MySQL (`portfolio_snapshots`)
```

---

## 5. Exhaustive Directory Specification

Every single folder below already exists, contains an `__init__.py`, and must have purposeful Python code implemented into it:

### 5.1. Configuration (`tapewright-quant/config/`)
- `environments/`: Environment variable loaders (`.env`), dev/prod run modes, log verbosity configs.
- `trading/`: Asset universe definitions, trading calendar (NYSE/crypto 24/7), risk constraints (max position size, max leverage, stop-loss thresholds).
- `database/`: MySQL credentials loader, connection pooling configs, retry logic.

### 5.2. Database Layer (`tapewright-quant/database/`)
- `schema/`: MySQL DDL scripts (`schema.sql` already provided).
- `migrations/`: Versioned migration runners written in pure Python.
- `connection/`: Singleton MySQL connection manager and context managers (`get_db_connection()`).
- `repositories/order_repo/`: CRUD operations for order entries and fill status.
- `repositories/portfolio_repo/`: Daily equity curve and drawdown logging to MySQL.
- `repositories/signal_repo/`: Signal persistence and historical signal retrieval.
- `repositories/market_data_repo/`: Metadata lookup for asset classes, tickers, and trading status.

### 5.3. Data Ingestion & Preprocessing (`tapewright-quant/data/`)
- `ingestion/market_feeds/equities/`: Yahoo Finance / Alpaca market data downloaders and generators.
- `ingestion/market_feeds/crypto/`: CCXT / Binance public REST/WebSocket tick fetchers.
- `ingestion/market_feeds/fx_forex/`: Forex currency pair rate fetchers.
- `ingestion/news_feeds/sec_edgar/`: SEC EDGAR 10-K/10-Q parser for corporate fundamentals.
- `ingestion/news_feeds/financial_rss/`: Financial news RSS scraper and timestamp normalizer.
- `ingestion/macroeconomic/fred_rates/`: Federal Reserve Economic Data (FRED) rate fetcher (CPI, Fed Funds Rate, Yield Curve).
- `cleaning/outlier_filters/`: Bad tick scrubbers, price spike detection, MAD (Median Absolute Deviation) filters.
- `cleaning/corporate_actions/`: Stock split and dividend adjustment math.
- `validation/schema_rules/`: Schema validation ensuring no missing values, negative prices, or timestamp gaps.
- `transformation/resamplers/`: Bar aggregation (tick to 1m, 5m, 1h, daily bars), volume bars, dollar bars.
- `caching/parquet_store/`: High-speed local Parquet reader/writer with partitioning by symbol and year.

### 5.4. Feature Engineering (`tapewright-quant/features/`)
- `technical_indicators/momentum/`: RSI, MACD, Stochastic Oscillator, Rate of Change (ROC).
- `technical_indicators/volatility/`: Average True Range (ATR), Bollinger Bands, Keltner Channels, Historical Volatility.
- `technical_indicators/trend/`: EMA, SMA, Supertrend, ADX (Average Directional Index).
- `technical_indicators/volume/`: On-Balance Volume (OBV), Volume-Weighted Average Price (VWAP), Chaikin Money Flow.
- `market_microstructure/order_book/`: Bid-ask spread estimators (Roll, Corwin-Schultz), order book imbalance (OBI).
- `market_microstructure/tick_entropy/`: Shannon entropy on price directional runs, trade flow toxicity (VPIN).
- `econometric/cointegration/`: Augmented Engle-Granger and Johansen tests for cointegrated asset pairs.
- `econometric/fractional_diff/`: Marcos López de Prado's fractional differentiation algorithm (stationarity with memory retention).
- `econometric/kalman_filter/`: Dynamic hedge ratio estimator via state-space Kalman filtering.
- `sentiment/lexicon_scoring/`: Loughran-McDonald financial dictionary sentiment analyzer.
- `sentiment/embeddings/`: Transformer text embeddings for financial headlines.
- `store/registry/`: Feature catalog for point-in-time feature retrieval (preventing lookahead bias).

### 5.5. Predictive & ML Models (`tapewright-quant/models/`)
- `time_series/arima/`: Auto-ARIMA fitting, residual diagnosis (Ljung-Box test).
- `time_series/garch/`: GARCH(1,1) and EGARCH volatility forecasting.
- `time_series/prophet/`: Additive trend and seasonal forecasting.
- `time_series/state_space/`: Unobserved Components Models (UCM).
- `deep_learning/lstm/`: Multi-feature LSTM for directional price prediction.
- `deep_learning/temporal_fusion_transformer/`: Multi-horizon interpretable time-series transformer.
- `deep_learning/wavenet/`: Dilated causal 1D convolutional neural network for financial signals.
- `ensemble/xgboost/`: Gradient boosted trees for tabular cross-sectional return prediction.
- `ensemble/lightgbm/`: Fast gradient boosting for large indicator feature sets.
- `ensemble/catboost/`: Handling categorical sector/asset metadata without manual one-hot encoding.
- `reinforcement_learning/agents/`: PPO (Proximal Policy Optimization) and DQN trading agents.
- `reinforcement_learning/environments/`: OpenAI Gymnasium trading environment with action space (Buy/Sell/Hold/Weight) and realistic execution state.
- `reinforcement_learning/rewards/`: Differential Sharpe Ratio reward, Sortino reward, drawdown penalties.

### 5.6. Backtesting & Portfolio Management (`tapewright-quant/backtesting/`)
- `engine/event_loop/`: Discrete event-driven backtesting engine (TickEvent, SignalEvent, OrderEvent, FillEvent) avoiding vectorized lookahead bias.
- `execution/slippage_models/`: Fixed basis point, linear volume impact, and square-root Kyle lambda slippage models.
- `execution/transaction_costs/`: Commission models (Maker/Taker fees, SEC exchange fees).
- `portfolio/markowitz/`: Mean-Variance Optimization, Modern Portfolio Theory efficient frontier generator.
- `portfolio/black_litterman/`: Bayesian view blending with market equilibrium weights.
- `portfolio/risk_parity/`: Hierarchical Risk Parity (HRP) using dendrogram clustering on asset covariance.
- `metrics/risk_measures/`: Annualized return, Sharpe, Sortino, Calmar, Value at Risk (VaR), Conditional VaR (CVaR).
- `metrics/drawdown_analysis/`: Underwater curves, max drawdown duration, recovery factor.
- `reports/tearsheet/`: Complete strategy performance summary generator (text and ASCII equity curves).

### 5.7. Signals & Arbitrage (`tapewright-quant/signals/`)
- `generation/crossover/`: Dual EMA / Golden cross signal logic.
- `generation/breakout/`: Donchian channel and volatility breakout generators.
- `generation/mean_reversion/`: Bollinger Band / RSI extreme mean-reversion generators.
- `filtering/volatility_filter/`: Regime filters blocking trades during extreme GARCH volatility spikes.
- `ranking/cross_sectional/`: Cross-sectional momentum / value decile rankers.
- `arbitrage/pairs_trading/`: Z-score spread thresholds on cointegrated pairs.

### 5.8. Pipelines, Utilities & Testing (`tapewright-quant/pipelines/`, `utils/`, `tests/`)
- `pipelines/etl/`: Ingestion to feature store orchestrator.
- `pipelines/training/`: Walk-forward validation and purged k-fold cross-validation pipeline.
- `pipelines/inference/`: Daily signal generation batch pipeline.
- `pipelines/monitoring/`: Concept drift and model performance decay monitor.
- `utils/math/`: Linear algebra helpers, covariance shrinkage (Ledoit-Wolf).
- `utils/dates/`: Trading holidays, timestamp conversions, business day delta calculators.
- `utils/profiling/`: Execution time decorators and memory footprint analyzers.
- `utils/logging/`: Structured console and file loggers.
- `tests/unit/`: Pytest tests for data, indicators, math, and models.
- `tests/integration/`: End-to-end simulation tests verifying zero lookahead bias and accurate MySQL record insertion.

---

## 6. Recommended Step-by-Step Implementation Roadmap for Claude

When starting work on this project, follow this prioritized sequence:

1. **Step 1: Configuration & Database Connector**
   - Populate `config/` and `database/connection/` with a clean, thread-safe MySQL connector and SQLite/mock fallback for automated unit test runs without requiring a live server.
2. **Step 2: Synthetic / Real Market Data Ingestion & Parquet Cache**
   - Implement `data/ingestion/` (with a built-in synthetic geometric Brownian motion / jump-diffusion market data generator for instant testing) and `data/caching/parquet_store/`.
3. **Step 3: Core Feature Engineering**
   - Populate `features/technical_indicators/` and `features/econometric/fractional_diff/` using vectorized NumPy/Pandas.
4. **Step 4: Event-Driven Backtesting Core**
   - Implement `backtesting/engine/event_loop/` with proper event hierarchy (`TickEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`).
5. **Step 5: Strategy Signals & Portfolio Optimization**
   - Build `signals/generation/` and `backtesting/portfolio/risk_parity/` and `backtesting/metrics/risk_measures/`.
6. **Step 6: Machine Learning & Time Series Models**
   - Implement `models/time_series/garch/` and `models/ensemble/xgboost/` with purged walk-forward cross-validation.
7. **Step 7: Verification & Test Suite**
   - Run tests in `tests/` ensuring all modules interoperate cleanly with 100% Python syntax validity.
