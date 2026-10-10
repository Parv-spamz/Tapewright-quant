# Tapewright-Quant: Institutional Quant Research & Execution Engine

Tapewright-Quant is an institutional-grade algorithmic research and execution platform built on rigorous mathematical econometrics, market microstructure simulation, predictive causality gating, and anti-overfitting statistics.

Unlike retail backtesting libraries that assume frictionless fills, suffer from lookahead leakage, and report inflated Sharpe ratios, Tapewright-Quant models the physical limit order book, enforces causal temporal boundaries, purges overlapping cross-validation labels, and penalizes multi-trial strategy selection.

---

## The 6 Pillars & Proving Tests

| Pillar | Description | Proving Test in Test Suite |
| :--- | :--- | :--- |
| **1. Microstructure-Realistic Execution** | Simulates Level-2 order book queue priority, Hasbrouck lambda square-root price impact, Kyle's lambda, and Volume-Synchronized Probability of Toxicity (VPIN). Orders are filled causally strictly after signal generation. | `tests/unit/test_execution.py`<br>`tests/unit/test_synthetic_clean.py::TestL2TapeSimulation` |
| **2. Marcos López de Prado Econometrics** | Fractional Differentiation (FFD) preserving memory at minimum stationary $d$, Triple-Barrier labeling with vertical/horizontal barriers, secondary meta-labeling for position sizing, and Purged/Embargoed Cross-Validation. | `tests/unit/test_econometrics.py`<br>`tests/unit/test_labeling_cv.py`<br>`tests/unit/test_training.py::TestPurgedCVHonesty` |
| **3. Predictive-Causality Gating** | Rejects spurious correlations by filtering signals through bivariate Granger causality and non-parametric Transfer Entropy (TE) with Benjamini-Hochberg False Discovery Rate (FDR) $q$-value step-up control. | `tests/unit/test_econometrics.py::TestCausalGraphIntegration`<br>`tests/unit/test_signals_portfolio.py::TestCausalGate` |
| **4. Half-Kelly + CVaR Risk Budgeting** | Position sizing governed by Half-Kelly leverage scaling, Cornish-Fisher Value-at-Risk (VaR), and parametric/historical Conditional Value-at-Risk (CVaR) loss limits, coupled with dynamic volatility targeting. | `tests/unit/test_signals_portfolio.py::TestPortfolioAllocation`<br>`tests/unit/test_pipelines_stress.py` |
| **5. Anti-Overfitting Statistics** | Computes the Deflated Sharpe Ratio (DSR) correcting for non-normality and selection bias ($N$ trials), and Combinatorially Symmetric Cross-Validation (CSCV) Probability of Backtest Overfitting (PBO). | `tests/unit/test_metrics_overfit.py`<br>`tests/integration/test_overfit_audit.py` |
| **6. Stress Testing & Interactive Dashboard** | Factor scenario replay (Lehman 2008, COVID 2020), reverse stress test solver identifying break-even market shocks, and an interactive Streamlit & static HTML tearsheet dashboard. | `tests/unit/test_pipelines_stress.py`<br>`tests/integration/test_pipeline_e2e.py` |

---

## Architectural Map

```
                                  +---------------------------------------+
                                  |     MARKET DATA & ORDER BOOK L2       |
                                  |  (Synthetic L2 / Parquet Bar Store)   |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |         FEATURE ENGINEERING           |
                                  |  * Fractional Diff FFD (Memory > 90%) |
                                  |  * L2 Imbalance / Roll / Hasbrouck    |
                                  |  * Microstructure Noise & Realized Vol|
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |       PREDICTIVE CAUSALITY GATE       |
                                  |  * Granger Causality Tests            |
                                  |  * Transfer Entropy (Permutation)     |
                                  |  * Benjamini-Hochberg FDR Control     |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |       ALPHA & ML MODEL PIPELINES      |
                                  |  * Primary Signals (Crossover/Donchian|
                                  |  * Ensembles (XGBoost/LightGBM/CatB)  |
                                  |  * Meta-Labeling & Purged K-Fold CV   |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |     RISK BUDGETING & PORTFOLIO        |
                                  |  * Half-Kelly Bet Sizing              |
                                  |  * Hierarchical Risk Parity (HRP)     |
                                  |  * CVaR Constraints & Vol Target      |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |      REALISTIC MATCHING ENGINE        |
                                  |  * Queue Priority & Latency Delay     |
                                  |  * Square-root Price Impact           |
                                  |  * VPIN Adverse Selection Monitoring  |
                                  +-------------------+-------------------+
                                                      |
                         +----------------------------+----------------------------+
                         v                                                         v
         +-------------------------------+                         +-------------------------------+
         |   AUDIT & OVERFITTING SUITE   |                         |      ANALYTICS & DASHBOARD    |
         |  * Deflated Sharpe Ratio (DSR)|                         |  * Streamlit Interactive App  |
         |  * PBO via CSCV Partitions    |                         |  * Static HTML Tearsheet      |
         |  * Zero-Lookahead Audit Test  |                         |  * MySQL / SQLite Persistence |
         +-------------------------------+                         +-------------------------------+
```

---

## Installation & Setup

### Environment Requirements
- **Python**: 3.10+ (tested on Python 3.14)
- **Database**: MySQL 8.0+ (optional in development; auto-defaults to SQLite)
- **Dependencies**: numpy, scipy, pandas, pyarrow, statsmodels, arch, scikit-learn, torch, xgboost, lightgbm, catboost, gymnasium, pymysql, pytest, streamlit, plotly.

```bash
# Clone repository
git clone <repo-url> tapewright-quant
cd tapewright-quant

# Install Python requirements
pip install -r requirements.txt
```

---

## Running the Engine

### 1. Run Complete End-to-End Demo Simulation
Runs synthetic data generation, feature engineering, causality gating, meta-label training, backtesting, stress testing, and generates report artifacts in `reports_out/`:
```bash
python -m pipelines.run_demo
```

### 2. Launch the Interactive Streamlit Tearsheet Dashboard
Launches the interactive local web dashboard for portfolio visual diagnostics, cumulative equity curves, drawdown waterfalls, trade execution logs, and stress test shocks:
```bash
streamlit run backtesting/reports/tearsheet/app.py
```

### 3. Run the Complete Test Suite
Execute the entire test suite (221 passing unit/integration tests):
```bash
python -m pytest -v
```

### 4. Run Audits & Constraint Verification
Verify syntax across 235 files, zero-lookahead invariance, and confirm strict compliance with zero-TypeScript constraints:
```bash
python scripts/check_syntax.py
python scripts/check_constraints.py
python -m compileall -q .
```

---

## Database Configuration

Tapewright-Quant uses a relational database strictly for metadata and accounting tracking across 5 standard tables:
- `assets`
- `model_registry`
- `trade_signals`
- `order_executions`
- `portfolio_snapshots`

High-frequency ticks, bars, and feature matrices reside strictly in Parquet storage.

### MySQL Environment Variables (Optional)
To connect to an active MySQL production instance, export the following environment variables:
```bash
export TAPEWRIGHT_DB_HOST="localhost"
export TAPEWRIGHT_DB_PORT="3306"
export TAPEWRIGHT_DB_USER="tapewright_admin"
export TAPEWRIGHT_DB_PASSWORD="your_secure_password"
export TAPEWRIGHT_DB_NAME="tapewright_db"
```
When `TAPEWRIGHT_DB_HOST` is unset, the engine automatically uses an isolated SQLite file/in-memory database with identical relational schema migrations.

---

## Repository Map

```
tapewright-quant/
├── backtesting/               # Simulation, realistic matching engine, metrics, reports
│   ├── engine/                # Event-driven backtester & order queue execution loop
│   ├── execution/             # L2 queue position, market impact, slippage, and fees
│   ├── metrics/               # Sharpe, Sortino, Drawdown, DSR, and CSCV PBO
│   └── reports/               # HTML tearsheet generator & Streamlit dashboard
├── config/                    # Configuration dataclasses & YAML loader
├── database/                  # Schema migrations, connection pool, and repositories
│   ├── connection/            # PyMySQL / SQLite manager with transaction context
│   ├── migrations/            # Versioned SQL migrations (001_initial.sql)
│   ├── repositories/          # CRUD repositories for signals, orders, snapshots, models
│   └── schema/                # Canonical MySQL DDL schema (schema.sql)
├── data/                      # Market data ingest, cleaning, and storage
│   ├── caching/               # Partitioned Parquet data store
│   └── ingestion/             # Synthetic GBM, Jump-Diffusion, Heston, and L2 generators
├── features/                  # Econometrics, order book micro, and labeling
│   ├── econometric/           # Fractional differentiation (FFD), cointegration, Kalman
│   ├── labeling/              # Triple-barrier method & daily volatility
│   └── market_microstructure/ # L2 depth imbalance, Hasbrouck lambda, Roll spread, VPIN
├── models/                    # Predictive alpha models & registry
│   ├── deep_learning/         # PyTorch SimpleTFT, WaveNet causal convs, LSTM
│   ├── ensemble/              # XGBoost, LightGBM, and CatBoost tabular models
│   ├── reinforcement/         # Gymnasium trading environment, DQN, and PPO
│   └── time_series/           # ARIMA, GARCH(1,1), AdditiveTrendSeasonal, State-space
├── pipelines/                 # Training, walk-forward, stress testing, and demo
│   ├── run_demo.py            # Master end-to-end execution script
│   ├── stress_testing/        # Multi-factor scenario replay & reverse stress tests
│   └── training/              # PurgedKFold CV, meta-labeling, and walk-forward signals
├── reports_out/               # Generated HTML dashboard, CSV tearsheets, AUDIT.md
├── scripts/                   # Auditing scripts (check_syntax.py, check_constraints.py)
└── tests/                     # 222-item unit & integration verification test suite
    ├── integration/           # Zero-lookahead, DB records audit, PBO/DSR overfit audit
    └── unit/                  # Pillar-by-pillar isolated mathematical proofs
```

---

## Architectural Constraints

- **Languages**: Python 3.10+ and MySQL exclusively.
- **Zero TypeScript / JavaScript**: Strict enforcement: no `.ts`, `.tsx`, `.js`, `package.json`, or node toolchains.
- **Strict Allowed Libraries**: `numpy`, `scipy`, `pandas`, `pyarrow`, `statsmodels`, `arch`, `scikit-learn`, `torch`, `xgboost`, `lightgbm`, `catboost`, `gymnasium`, `pymysql`, `pytest`, `streamlit`, `plotly`.
- **Zero Lookahead Guarantee**: Tested across all strategies ($N=300$ vs $N=400$ equity invariance to $10^{-9}$ tolerance).
- **Accounting Invariant**: Total portfolio NAV is strictly identical to $\text{cash} + \sum (\text{positions} \times \text{price})$ to $10^{-6}$ precision across all simulation cycles.

---

## Known Limitations

*(Refer to `reports_out/AUDIT.md` for full audit details and empirical metrics)*
1. **Mathematical Simulation vs Live Execution**: The order book simulator models queue mechanics and Hasbrouck lambda price impact, but live exchange execution entails non-deterministic exchange latency jitter and routing dynamics.
2. **Deterministic CPU Sizing**: Deep learning neural network parameters are scaled for rapid, reproducible CI verification on CPU.
3. **Dictionary Sentiment Analysis**: Sentiment scoring relies on domain-specific stem lexicons and TF-IDF features rather than heavyweight cloud-based LLM APIs.
4. **Synthetic Data Disclaimer**: Backtests conducted on synthetic data provide algorithmic verification, but say nothing about real-market profitability.
