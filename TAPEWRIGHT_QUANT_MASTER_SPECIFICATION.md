# Tapewright-Quant (FinPulse-Apex): Complete Architectural Specification & Quant System Master Guide

> **Document Classification**: Master System Specification, Mathematical Foundations & Institutional Evidence Dossier  
> **Target Audience / Ingestion Target**: ChatGPT / Advanced AI Reasoning Models / Quantitative Research Committees  
> **System Name**: Tapewright-Quant (Research Identity: FinPulse-Apex)  
> **Core Identity**: Institutional-Grade, Strictly Causal, Microstructure-Realistic Discrete-Event Quantitative Trading & Cryptographic Evidence Engine  
> **Implementation Tech Stack**: Python 3.10+ (pure standard library + curated scientific stack), MySQL 8.0+ / SQLite dual relational persistence, Parquet columnar time-series storage, Streamlit & Plotly interactive visualization. (Zero Node.js, Zero TypeScript, Zero C++/Rust compilation baggage, Zero cloud DB lock-in).

---

## Table of Contents
1. [Executive Summary & The "Why": How This Destroys 99% of GitHub Quant Repos](#1-executive-summary--the-why)
2. [The 5 Institutional Core Pillars & Differentiators](#2-the-5-institutional-core-pillars--differentiators)
   - [Pillar 1: Microstructure Realism & Synthetic Level-2 LOB Matching](#pillar-1-microstructure-realism--synthetic-level-2-lob-matching)
   - [Pillar 2: Marcos López de Prado Financial Econometrics](#pillar-2-marcos-lópez-de-prado-financial-econometrics)
   - [Pillar 3: Predictive Causal Graph Discovery & Spurious Alpha Gating](#pillar-3-predictive-causal-graph-discovery--spurious-alpha-gating)
   - [Pillar 4: Dynamic Half-Kelly & CVaR Portfolio Risk Budgeting](#pillar-4-dynamic-half-kelly--cvar-portfolio-risk-budgeting)
   - [Pillar 5: Macro Shock Stress-Testing & Interactive Quant Analytics](#pillar-5-macro-shock-stress-testing--interactive-quant-analytics)
3. [The Trust Layer: Cryptographic Ledger & Anti-Overfitting Safeguards](#3-the-trust-layer-cryptographic-ledger--anti-overfitting-safeguards)
4. [End-to-End System Architecture & Data Flow](#4-end-to-end-system-architecture--data-flow)
5. [Exhaustive Subsystem & Module Breakdown](#5-exhaustive-subsystem--module-breakdown)
   - [5.1 Configuration & Risk Engine (`config/`)](#51-configuration--risk-engine-config)
   - [5.2 Minimalist Relational Database Layer (`database/`)](#52-minimalist-relational-database-layer-database)
   - [5.3 Data Ingestion, Synthesis, Cleaning & Parquet Storage (`data/`)](#53-data-ingestion-synthesis-cleaning--parquet-storage-data)
   - [5.4 Feature Engineering & Microstructure Quant Library (`features/`)](#54-feature-engineering--microstructure-quant-library-features)
   - [5.5 Alpha Models, Deep Learning & Reinforcement Learning (`models/`)](#55-alpha-models-deep-learning--reinforcement-learning-models)
   - [5.6 Strictly Causal Discrete-Event Backtester (`backtesting/`)](#56-strictly-causal-discrete-event-backtester-backtesting)
   - [5.7 Alpha Signal Generation, Regime Filters & Arbitrage (`signals/`)](#57-alpha-signal-generation-regime-filters--arbitrage-signals)
   - [5.8 Orchestration Pipelines & Drift Monitoring (`pipelines/`)](#58-orchestration-pipelines--drift-monitoring-pipelines)
   - [5.9 Research Protocols, Ablation & Microstructure Calibration (`research/`)](#59-research-protocols-ablation--microstructure-calibration-research)
   - [5.10 Cryptographic Integrity & Automated Verification Gate (`integrity/`)](#510-cryptographic-integrity--automated-verification-gate-integrity)
   - [5.11 Mathematical Utilities & Core Helpers (`utils/`)](#511-mathematical-utilities--core-helpers-utils)
6. [Verification, Hardening & Quality Invariants](#6-verification-hardening--quality-invariants)
7. [Empirical Research Findings & The Scientific Truth](#7-empirical-research-findings--the-scientific-truth)
8. [Comprehensive Directory & File Map](#8-comprehensive-directory--file-map)
9. [Briefing Prompt for ChatGPT](#9-briefing-prompt-for-chatgpt)

---

## 1. Executive Summary & The "Why"

### The Problem with 99% of Open-Source "Quant Trading" Repositories
The internet is flooded with thousands of repositories labeled "Trading Bot", "AI Quant", or "Algorithmic Trader". Virtually all of them suffer from fatal amateur flaws:
1. **Vectorized Lookahead Bias**: They compute `df['signal'] = df['close'].rolling(...)` followed by `df['returns'] = df['signal'] * df['close'].pct_change()`. This peeks into future prices, assumes trades fill instantaneously at the close price of the signal bar, and produces absurd backtests showing 500% annualized returns that completely collapse in live execution.
2. **Obsolete 1970s Heuristics**: They rely on uncalibrated 14-period RSI or MACD crossovers—signals whose alpha decayed to zero decades ago in modern electronic markets.
3. **Severe Cross-Validation Leakage**: Standard random $k$-fold cross-validation or un-embargoed train/test splits leak future auto-regressive information and overlapping event labels into training sets, producing hyper-inflated, completely bogus test scores.
4. **Zero Microstructure Realism**: They assume 100% infinite liquidity, zero spread, zero adverse selection, zero order queue latency, and zero market impact.
5. **Data Mining & Selection Bias**: Authors run 500 parameter iterations, cherry-pick the single random run that got lucky during a bull run, and present it as an "alpha strategy".

### What Tapewright-Quant (FinPulse-Apex) Does Instead
Tapewright-Quant was engineered from first principles to reflect the mathematical, operational, and statistical rigor of premier quantitative hedge funds (Two Sigma, Citadel, Renaissance Technologies, AQR).

- **100% Strictly Causal Discrete-Event Engine**: Discrete events (`TickEvent` $\to$ `SignalEvent` $\to$ `OrderEvent` $\to$ `FillEvent`). A signal generated at the close of bar $t$ can only be submitted for execution at bar $t+1$, filling at $t+1$ open price with simulated latency and price impact. Zero future peeking is mathematically guaranteed.
- **Microstructure-Realistic Execution**: Features a simulated Level-2 Limit Order Book (LOB) with Poisson arrival queues, queue cancellation decay, Corwin-Schultz & Roll bid-ask spread estimators, and Kyle's Lambda / Hasbrouck square-root market impact modeling.
- **Marcos López de Prado Econometrics**: Features fractional differentiation to preserve multi-scale price memory while achieving stationarity, triple-barrier labeling, purged & embargoed cross-validation, and secondary meta-labeling with Half-Kelly position sizing.
- **Predictive Causal Graph Discovery**: Uses Granger causality F-tests and non-parametric Transfer Entropy with False Discovery Rate (FDR) control to prune spurious correlations before signals can ever reach production models.
- **Cryptographic Provenance Audit Ledger**: Every dataset checksum, parameter trial, backtest spec, metric calculation, and forward emission is hashed (SHA-256) and chained into an immutable audit ledger.
- **Brutal Scientific Honesty & Protocol Pre-Registration**: When evaluated under strict pre-registered rules (Deflated Sharpe Ratio $\ge 0.90$, validation Sharpe $> 0$, Probabilistic Sharpe Ratio $\ge 0.90$), zero candidate models survived on held-out data. Instead of tweaking thresholds to manufacture fake wins, the platform records this empirical reality transparently.
- **Offline Deterministic Reproducibility**: Content-addressed raw data archives rebuild all datasets without network access, reproducing baseline and trial results to $10^{-9}$ numerical precision.

---

## 2. The 5 Institutional Core Pillars & Differentiators

```
+----------------------------------------------------------------------------------------------------+
|                                    TAPEWRIGHT-QUANT ARCHITECTURE                                   |
|                                                                                                    |
|   +--------------------------------------------------------------------------------------------+   |
|   | PILLAR 1: Microstructure Realism (L2 Queue, Kyle Lambda, Hasbrouck Impact, VPIN, Spreads)  |   |
|   +--------------------------------------------------------------------------------------------+   |
|                                                 ▲                                                  |
|                                                 │                                                  |
|   +--------------------------------------------------------------------------------------------+   |
|   | PILLAR 2: López de Prado Econometrics (FFD Memory, Triple Barrier, Purged CV, Meta-Labels) |   |
|   +--------------------------------------------------------------------------------------------+   |
|                                                 ▲                                                  |
|                                                 │                                                  |
|   +--------------------------------------------------------------------------------------------+   |
|   | PILLAR 3: Causal Graph Discovery (Granger Causality, Transfer Entropy, FDR Benjamini-Hoch) |   |
|   +--------------------------------------------------------------------------------------------+   |
|                                                 ▲                                                  |
|                                                 │                                                  |
|   +--------------------------------------------------------------------------------------------+   |
|   | PILLAR 4: Risk Budgeting & Optimization (Half-Kelly, HRP, ERC, Ledoit-Wolf, 99% CVaR)      |   |
|   +--------------------------------------------------------------------------------------------+   |
|                                                 ▲                                                  |
|                                                 │                                                  |
|   +--------------------------------------------------------------------------------------------+   |
|   | PILLAR 5: Macro Stress Calibration (2008 Lehman, 2020 COVID, 2022 Hikes, Reverse Stress)  |   |
|   +--------------------------------------------------------------------------------------------+   |
+----------------------------------------------------------------------------------------------------+
```

### Pillar 1: Microstructure Realism & Synthetic Level-2 LOB Matching
In real institutional trading, execution costs and order book microstructure dominate strategy PnL. Tapewright-Quant avoids toy fixed basis-point slippage assumptions by deploying realistic market friction mechanics:
1. **Level-2 Limit Order Book Simulation (`LOBSimulator`)**:
   - Maintains explicit multi-level price queues (5 bids, 5 asks) with stochastic Poisson order arrivals and cancellations.
   - Orders entering the book join the tail of the queue at that price level.
   - Fills only occur if the cumulative matching trading volume exceeds the quantity of orders queued ahead of the participant (`ahead_qty`).
2. **Empirical Kyle's Lambda & Square-Root Market Impact**:
   Price impact $\Delta P$ is non-linear and scales with order volume relative to Average Daily Volume (ADV) and market volatility:
   $$\Delta P = c \cdot \sigma \cdot \left(\frac{Q}{\text{ADV}}\right)^\gamma$$
   Where $c = 0.1196$ and $\gamma = 1.1988$ (calibrated directly from empirical Binance high-frequency order book data with $R^2 = 0.4811$).
3. **High-Frequency Spread Estimators**:
   - **Roll (1984) Effective Spread**: Derives effective spread from return autocovariance: $S = 2\sqrt{-\text{Cov}(\Delta P_t, \Delta P_{t-1})}$.
   - **Corwin-Schultz (2012) High-Low Spread**: Estimates bid-ask spread from high-to-low price ratios over 1-day and 2-day windows, stripping out geometric drift.
4. **Adverse Selection & VPIN**:
   Calculates Volume-Synchronized Probability of Toxicity (VPIN) across volume buckets:
   $$\text{VPIN} = \frac{\sum_{\tau=1}^N |V_\tau^B - V_\tau^S|}{N \cdot V}$$
   Trades executed during high toxicity regimes incur dynamic adverse selection slippage surcharges.

---

### Pillar 2: Marcos López de Prado Financial Econometrics
Financial machine learning fails when textbook data science assumptions are blindly applied to non-stationary, auto-correlated financial series. Tapewright-Quant implements the complete econometric methodology from Marcos López de Prado's *Advances in Financial Machine Learning*:

1. **Fractional Differentiation (FFD)**:
   Standard integer differentiation ($d=1$, price returns) removes all long-memory properties, destroying the macro trends and price levels that generate alpha. Fractional differentiation computes binomial series expansions of $(1 - B)^d$:
   $$(1 - B)^d = \sum_{k=0}^\infty (-1)^k \binom{d}{k} B^k = 1 - d B + \frac{d(d-1)}{2!} B^2 - \frac{d(d-1)(d-2)}{3!} B^3 + \dots$$
   By finding the minimum degree $d^* \in [0.25, 0.50]$ that passes the Augmented Dickey-Fuller (ADF) test ($p < 0.05$), the engine achieves stationarity while preserving over **91.3% correlation with original price memory**.

2. **Dynamic State-Space Kalman Filter Pairs Trading**:
   Rather than static OLS cointegration regressions that break when cointegration vectors drift, Tapewright-Quant runs a 2-state Kalman filter estimating time-varying hedge ratios $\beta_t$ per bar:
   $$y_t = \alpha_t + \beta_t x_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, R)$$
   $$\begin{bmatrix} \alpha_t \\ \beta_t \end{bmatrix} = \begin{bmatrix} \alpha_{t-1} \\ \beta_{t-1} \end{bmatrix} + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(0, \mathbf{Q})$$
   Spreads are dynamically standardized into Z-scores: $Z_t = \frac{e_t - \mu_e}{\sigma_e}$, triggering mean-reversion trades with dynamic hedge weighting.

3. **Triple-Barrier Labeling**:
   Fixed-horizon return labeling ($y_t = P_{t+h} - P_t$) ignores paths taken by prices. Triple-barrier labeling sets:
   - **Upper Barrier**: Dynamic profit-taking target ($P_t + k \cdot \sigma_t$).
   - **Lower Barrier**: Dynamic stop-loss boundary ($P_t - k \cdot \sigma_t$).
   - **Vertical Barrier**: Time-out expiration ($t + \tau$).
   Labels reflect which barrier is touched first, matching real-world trading stops and profit exits.

4. **Purged & Embargoed Cross-Validation (`PurgedKFold`)**:
   Standard K-Fold leaks information because overlapping triple-barrier label intervals $[t_{i,0}, t_{i,1}]$ span across train and test boundaries. `PurgedKFold` purges training observations overlapping with test labels and places an embargo period after test folds to prevent auto-regressive information leakage.
   *Empirical Audit*: On synthetic noise labels, leaky K-Fold produces an inflated AUC of **0.6467**, whereas Purged CV produces an honest **0.4426**.

5. **Secondary Meta-Labeling**:
   Decouples the trading process into two distinct models:
   - **Primary Model**: High-recall heuristic or econometric model determining directional side ($+1$ or $-1$).
   - **Secondary ML Model**: Precision classifier trained on purged triple barriers predicting the probability of primary trade success ($P(\text{Success})$). Sizes positions via continuous Half-Kelly betting fractions.

---

### Pillar 3: Predictive Causal Graph Discovery & Spurious Alpha Gating
Machine learning algorithms are notoriously susceptible to fitting spurious correlations in financial data that invert violently out-of-sample. Tapewright-Quant incorporates a rigorous causal gatekeeper:

1. **Granger Causality F-Tests**:
   Tests whether historical lags of candidate feature $X$ provide statistically significant incremental predictive power for target asset return $Y$ beyond historical lags of $Y$ alone in a Vector Autoregression (VAR) framework.
2. **Non-Parametric Transfer Entropy**:
   Measures directed information transfer between non-linear processes without assuming Gaussian distributions:
   $$T_{X \to Y} = \sum p(y_{t+1}, y_t, x_t) \log \frac{p(y_{t+1} \mid y_t, x_t)}{p(y_{t+1} \mid y_t)}$$
   Permutation testing constructs empirical null distributions to test statistical significance ($p < 0.01$).
3. **Benjamini-Hochberg False Discovery Rate (FDR) Control**:
   Controls the expected proportion of false positive causal links across multi-feature screening ($q \le 0.05$). Achieves 100% topology recovery on planted synthetic causal DAGs.

---

### Pillar 4: Dynamic Half-Kelly & CVaR Portfolio Risk Budgeting
Traditional Markowitz Mean-Variance Optimization (MVO) creates extreme, highly leveraged, unstable portfolios that collapse when out-of-sample covariance matrices deviate from sample estimates. Tapewright-Quant replaces MVO with robust institutional portfolio engines:

1. **Hierarchical Risk Parity (HRP)**:
   Performs tree-clustering on asset covariance matrices via single-linkage dendrograms, quasi-diagonalizes the correlation matrix, and allocates weights using recursive bisection. Requires no matrix inversion, making it mathematically stable even for singular or ill-conditioned covariance matrices.
2. **Equal Risk Contribution (ERC) Risk Parity**:
   Optimizes portfolio weights such that the marginal risk contribution of every asset is equalized:
   $$w_i \cdot \frac{(\boldsymbol{\Sigma} \mathbf{w})_i}{\sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}} = w_j \cdot \frac{(\boldsymbol{\Sigma} \mathbf{w})_j}{\sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}} \quad \forall i, j$$
3. **Ledoit-Wolf Covariance Shrinkage**:
   Shrinks empirical sample covariance $\mathbf{S}$ toward a structured target $\mathbf{F}$ (constant correlation or identity) using an analytically optimal shrinkage intensity $\delta^*$:
   $$\boldsymbol{\Sigma}_{\text{shrunk}} = (1 - \delta^*) \mathbf{S} + \delta^* \mathbf{F}$$
4. **Dynamic Half-Kelly Sizing Clamped by 99% CVaR**:
   Sizes strategy exposures via continuous fractional Kelly:
   $$f^* = \frac{1}{2} \cdot \frac{\mu - r}{\sigma^2}$$
   Clamped dynamically by GARCH(1,1) conditional volatility forecasts and Cornish-Fisher Expected Shortfall (CVaR).

---

### Pillar 5: Macro Shock Stress-Testing & Interactive Quant Analytics
A robust quantitative strategy must be evaluated against structural macroeconomic crises:

1. **Empirical Historical Crisis Factor Replay**:
   Replays calibrated historical crisis shocks directly against the current portfolio:
   - **2008 Lehman GFC Collapse**: Realized Equity Shock = -46.04%, Volatility Multiplier = 2.86x
   - **March 2020 COVID-19 Flash Crash**: Realized Equity Shock = -33.72%, Volatility Multiplier = 5.57x
   - **2022 Federal Reserve Aggressive Rate Hike Cycle**: Realized Equity Shock = -24.50%, Volatility Multiplier = 1.89x
2. **Reverse Stress Testing**:
   Uses Newton-Raphson root-finding to solve for the exact minimal market factor shock required to breach a designated maximum portfolio loss threshold (e.g., -15% NAV).
3. **Dual Interactive Reporting Layer**:
   - **Interactive Multi-Tab Streamlit Dashboard (`app.py`)**: Features executive summary, underwater drawdown waterfalls, rolling VaR/CVaR, Monte Carlo 100-path confidence cones, trade logs, database inspectors, and live calibration evidence.
   - **Standalone Dark-Mode HTML Tearsheet (`dashboard.html`)**: Self-contained, responsive single-page institutional report with embedded interactive Plotly figures.

---

## 3. The Trust Layer: Cryptographic Ledger & Anti-Overfitting Safeguards

### Cryptographic Hash-Chained Audit Ledger
Every experiment, dataset, code module, and forward paper test signal is audited via an append-only cryptographic ledger (`research/ledger/` and `integrity/ledger.py`):
- Each ledger record is formatted as canonical JSON, hashed with SHA-256, and cryptographically chained to the previous record's hash (`prev_hash`).
- Tampering with any historical experiment metric, threshold, or dataset checksum immediately invalidates the entire hash chain downstream.
- External anchoring (`research/ledger/ANCHORS.txt`) logs ledger head hashes for verification against external immutable timestamp authorities (e.g., Git commit SHAs, PR merge logs, RFC 3161 timestamps).

```
+------------------------+      +------------------------+      +------------------------+
| Ledger Block #001      |      | Ledger Block #002      |      | Ledger Block #003      |
| tag: "dataset.raw"     |      | tag: "spec.experiment" |      | tag: "eval.metrics"    |
| prev_hash: "00000..."  | ───► | prev_hash: "a4f89..."  | ───► | prev_hash: "e7c12..."  |
| payload_hash: "..."    |      | payload_hash: "..."    |      | payload_hash: "..."    |
| entry_hash: "a4f89..." |      | entry_hash: "e7c12..." |      | entry_hash: "b901a..." |
+------------------------+      +------------------------+      +------------------------+
```

### Deflated Sharpe Ratio (DSR) & Combinatorially Symmetric Cross-Validation (CSCV)
In quantitative finance, evaluating multiple strategy variations on the same data leads to severe selection bias. Tapewright-Quant explicitly tracks the number of trials ($N=55$ on development split):
1. **Deflated Sharpe Ratio (DSR)**:
   Adjusts the observed Sharpe ratio for multi-testing selection bias, non-normal return distributions (skewness $\hat{\gamma}_3$ and kurtosis $\hat{\gamma}_4$), and track record length $T$:
   $$\text{DSR} = \Phi\left(\frac{(\widehat{\text{SR}} - \text{SR}_0)\sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \widehat{\text{SR}} + \frac{\hat{\gamma}_4 - 1}{4} \widehat{\text{SR}}^2}}\right)$$
   Where $\text{SR}_0$ is the expected maximum Sharpe ratio under the null hypothesis of no alpha across $N$ trials:
   $$\text{SR}_0 = \sqrt{2 \ln N} \left(1 - \frac{\gamma}{\ln N}\right) + \frac{\gamma}{\sqrt{2 \ln N}}$$
2. **CSCV Probability of Backtest Overfitting (PBO)**:
   Partitions historical data into $S=16$ slices, forms all $\binom{S}{S/2}$ combinatorial training and testing splits, ranks strategies in-sample, and calculates the empirical probability that the best in-sample strategy underperforms median performance out-of-sample ($PBO = 0.1580$).

### One-Shot Cryptographic Lockbox Holdout (`research/vault/`)
To protect against researcher overfitting:
- A pristine out-of-sample data split (`research/vault/`) is sealed and programmatically isolated.
- The evaluation CLI (`research/run_lockbox.py`) refuses to run without an explicit confirmation passphrase.
- The vault records an evaluation count (`eval_count=0`) and remains **UNOPENED**.

---

## 4. End-to-End System Architecture & Data Flow

```
[External Feeds / Microstructure Ingestion]
  Equities (Yahoo/Alpaca) | Crypto (CCXT/Binance L2) | Rates (FRED) | SEC Edgar
                             │
                             ▼
[Data Cleaning & Standardization Layer]
  Median Absolute Deviation (MAD) Spike Scrubber | Stock Split & Dividend Adjustments
  Schema Rule Validation | Dollar & Volume Bar Aggregation
                             │
                             ▼
[Columnar Parquet Store] (Partitioned: symbol=<SYM>/year=<YYYY>)
                             │
                             ▼
[Feature Engineering Engine]
  ├─ Momentum / Volatility / Trend / Volume Indicators
  ├─ Corwin-Schultz & Roll Spreads | Shannon Entropy | VPIN Toxicity
  ├─ Fractional Differentiation (FFD) | Dynamic Kalman Filter Cointegration
  └─ Causal Graph Discovery (Granger Causality + Transfer Entropy Gate)
                             │
                             ▼
[Alpha Prediction & Modeling Layer]
  ├─ Time Series: Auto-ARIMA | GARCH(1,1) Volatility | AdditiveTrendSeasonal
  ├─ Tabular Ensembles: XGBoost | LightGBM | CatBoost
  ├─ Deep Sequence: SimpleTFT (VSN + Multi-Quantile) | Dilated WaveNet | LSTM
  └─ Reinforcement Learning: Gymnasium TradingEnv | Double-DQN | PPO
                             │
                             ▼
[Signal Generation & Portfolio Sizing]
  ├─ Directional Signals: Crossovers, Breakouts, Mean-Reversion, Pairs Z-Score
  ├─ Lopez de Prado Meta-Labeling Classifier
  ├─ Dynamic Half-Kelly Bet Sizing Clamped by 99% CVaR
  └─ Portfolio Allocator: Hierarchical Risk Parity (HRP) | ERC | Ledoit-Wolf
                             │
                             ▼
[Relational Logging: MySQL / SQLite] ──► Audit: `trade_signals`
                             │
                             ▼
[Strictly Causal Discrete-Event Backtester]
  TickEvent (t) ──► SignalEvent (t) ──► OrderEvent (t) ──► FillEvent (t+1 open)
                             │
                             ├─ LOBSimulator: Level-2 Queue Position Matching
                             ├─ Kyle's Lambda & Hasbrouck Square-Root Impact
                             └─ Maker/Taker Exchange Fees & Clearing Slippage
                             │
                             ▼
[Portfolio Accounting & Invariant Verification] ──► Audit: `order_executions`
  Cash + Positions * Price == NAV (to 1e-6)     ──► Audit: `portfolio_snapshots`
                             │
                             ▼
[Reporting & Anti-Overfitting Diagnostics]
  ├─ Interactive Streamlit Dashboard (`app.py`)
  ├─ Dark-Mode HTML Tearsheet (`dashboard.html`)
  ├─ DSR & CSCV PBO Overfitting Diagnostics
  └─ Macro Factor Stress Replay (2008, 2020, 2022) & Reverse Stress Solver
```

---

## 5. Exhaustive Subsystem & Module Breakdown

### 5.1 Configuration & Risk Engine (`config/`)
- `environments/env_loader.py`: Validates environment settings (`.env`), database hosts, API keys, run modes (dev/prod), and log verbosity.
- `trading/universe.py`: Defines asset universes across liquid ETFs (`SPY`, `QQQ`, `IWM`, `EFA`, `TLT`, `IEF`, `GLD`, `XLE`, `XLF`, `XLK`, `XLV`, `XLY`), crypto pairs (`BTC/USDT`, `ETH/USDT`, `SOL/USDT`), and FX currency pairs.
- `trading/risk_limits.py`: Enforces institutional risk constraints: maximum gross leverage (e.g. 2.0x), maximum single-position weight (e.g. 20%), hard stop-loss thresholds, daily drawdown limits, and volatility targeting factors.
- `trading/calendar.py`: Manages NYSE equity market hours, holiday schedules, early market closes, and 24/7 continuous crypto market calendars.
- `database/db_config.py`: Configuration container and connection pooling parameters for MySQL 8.0+ and SQLite fallback.

### 5.2 Minimalist Relational Database Layer (`database/`)
Follows the **Minimal-DB Philosophy**: MySQL is used strictly for relational metadata, audit trails, and execution snapshots—never for high-frequency tick data arrays or large indicator matrices (which are stored in Parquet).
- `connection/connection_manager.py`: Thread-safe singleton connection manager supporting MySQL connection pools and offline zero-config SQLite. Enforces atomic transaction context managers (`get_db_connection()`) with automatic rollback on error.
- `schema/schema.sql`: Production MySQL DDL defining 6 core tables:
  1. `assets`: Master universe metadata (symbols, asset classes, exchanges, base currency, active status).
  2. `daily_market_summary`: High-level daily bar summaries and VWAP.
  3. `model_registry`: Machine learning and econometric model registry (hyperparameters, metrics, training timestamps).
  4. `trade_signals`: Generated signals with model IDs, timestamps, signal types (BUY/SELL/HOLD), confidence scores, and target weights.
  5. `order_executions`: Filled orders, execution timestamps, fill prices, slippage costs, and exchange fees.
  6. `portfolio_snapshots`: Daily mark-to-market NAV, cash balances, unrealized/realized PnL, rolling Sharpe, and max drawdown.
- `migrations/`: Pure Python versioned schema migration runners.
- `repositories/`: Clean repository pattern abstractions:
  - `market_data_repo/`: Asset metadata retrieval and active universe filters.
  - `signal_repo/`: Signal persistence and historical signal retrieval.
  - `order_repo/`: Order lifecycle logging, fill updates, and execution history.
  - `portfolio_repo/`: Daily equity curve and drawdown persistence.

### 5.3 Data Ingestion, Synthesis, Cleaning & Parquet Storage (`data/`)
- `ingestion/market_feeds/`: Ingestion connectors for Equities, CCXT crypto exchanges, and FX currency rates.
- `ingestion/macroeconomic/fred_rates/`: Federal Reserve Economic Data (FRED) fetcher for Treasury yields (10Y, 2Y yield curve inversion), CPI inflation, and Fed Funds Rates.
- `ingestion/news_feeds/sec_edgar/`: SEC EDGAR 10-K/10-Q fundamental filings parser.
- `ingestion/synthetic/`: Microstructure-realistic synthetic stochastic generators:
  - Merton Jump-Diffusion model: $\frac{dS_t}{S_t} = (\mu - \lambda k) dt + \sigma dW_t + J_t dq_t$ for capturing fat tails and crash risk.
  - Heston Stochastic Volatility model: simulates mean-reverting volatility with leverage effects.
  - Geometric Brownian Motion (GBM) with correlated multi-asset drift.
  - Synthetic Poisson Level-2 order book generator.
- `cleaning/outlier_filters/`: Bad tick scrubber using rolling Median Absolute Deviation (MAD) spike detection.
- `cleaning/corporate_actions/`: Stock split and cash dividend backwards adjustment math.
- `validation/schema_rules/`: Schema validation ensuring non-negative prices, monotonic timestamps, and zero missing values.
- `transformation/resamplers/`: Aggregates ticks into standard time bars (1m, 5m, 1h, daily), volume bars, and dollar bars.
- `caching/parquet_store.py`: High-throughput PyArrow columnar storage partitioned by `symbol=<SYM>/year=<YYYY>` enabling sub-millisecond multi-year slice reads.

### 5.4 Feature Engineering & Microstructure Quant Library (`features/`)
- `technical_indicators/`: Vectorized NumPy/Pandas implementations of Momentum (RSI, MACD, Stochastic Oscillator, ROC), Volatility (ATR, Bollinger Bands, Keltner Channels), Trend (EMA, SMA, Supertrend, ADX), Volume (OBV, VWAP, Chaikin Money Flow).
- `market_microstructure/order_book/spreads.py`:
  - Roll (1984) effective bid-ask spread estimator.
  - Corwin-Schultz (2012) high-low spread estimator.
  - Hasbrouck price impact lambda estimator.
  - Order Book Imbalance (OBI) across depth levels: $\text{OBI} = \frac{V_{\text{bid}} - V_{\text{ask}}}{V_{\text{bid}} + V_{\text{ask}}}$.
- `market_microstructure/tick_entropy/entropy.py`:
  - Shannon entropy over price run directions: $H(X) = -\sum p(x) \log_2 p(x)$.
  - VPIN (Volume-Synchronized Probability of Toxicity) across synchronized volume buckets.
- `econometric/fractional_diff/ffd.py`: Fast binomial expansion Fractional Differentiation with optimal $d^*$ solver via ADF stationarity tests.
- `econometric/cointegration/`:
  - Engle-Granger two-step regression and Johansen trace eigenvalue test.
  - Ornstein-Uhlenbeck mean-reversion half-life solver: $t_{1/2} = \frac{\ln 2}{\theta}$.
  - `causal_graph.py`: Granger causality and non-parametric Transfer Entropy directed acyclic graph (DAG) constructor with Benjamini-Hochberg FDR control.
- `econometric/kalman_filter/`: 2-state dynamic Kalman filter updating time-varying hedge ratios per bar.
- `sentiment/lexicon_scoring/`: Loughran-McDonald financial dictionary sentiment parser (scoring negative, positive, uncertainty, litigious, and constraining tones).
- `sentiment/embeddings/`: Transformer text embeddings for financial headlines.
- `store/registry/registry.py`: Point-in-time feature catalog enforcing strict chronological feature joins to eliminate lookahead leakage.

### 5.5 Alpha Models, Deep Learning & Reinforcement Learning (`models/`)
- `time_series/`:
  - `arima/`: Automated ARIMA model fitting with AIC grid optimization and Ljung-Box autocorrelation diagnostic testing on residuals.
  - `garch/`: GARCH(1,1) and EGARCH conditional volatility forecasting with Student's-$t$ innovation distributions.
  - `prophet/`: Additive trend and seasonal decomposition forecasting.
  - `state_space/`: Unobserved Components Models (UCM) separating cycle, trend, and noise.
- `ensemble/`: Leakage-safe tabular gradient boosted decision trees:
  - `xgboost/`, `lightgbm/`, `catboost/`: Production wrappers with sample-weighted loss, Purged Walk-Forward Cross Validation, and out-of-sample permutation feature importance.
- `deep_learning/`: Pure PyTorch, CPU-friendly sequence architectures:
  - `SimpleTFT`: Temporal Fusion Transformer featuring Gated Residual Networks (GRN), Variable Selection Networks (VSN) with attention weights, and multi-quantile pinball loss ($0.1, 0.5, 0.9$).
  - `WaveNetForecaster`: Dilated causal 1D convolutional neural network guaranteeing strict temporal receptive field boundaries.
  - `LSTMForecaster`: Multi-layer recurrent neural network for directional returns.
- `reinforcement_learning/`:
  - `environments/trading_env.py`: OpenAI Gymnasium-compliant trading environment featuring realistic execution states, cash accounting, order queue latencies, and slippage penalties.
  - `agents/`: Pure PyTorch Double-DQN (with experience replay buffer and target network) and clipped PPO (Proximal Policy Optimization).
  - `rewards/`: Differential Sharpe Ratio online rewards and drawdown penalty formulations.
- `registry.py`: Local model registry logging model artifacts, hyperparameter JSONs, and evaluation metrics.

### 5.6 Strictly Causal Discrete-Event Backtester (`backtesting/`)
- `engine/event_loop/`:
  - `events.py`: Event hierarchy (`TickEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`).
  - `data_handler.py`: `HistoricBarDataHandler` streaming chronological bars across multiple assets with deterministic symbol sorting.
  - `strategy_base.py`: Abstract strategy interface receiving `TickEvent` and emitting `SignalEvent`.
  - `execution.py`: Simulates broker fill mechanics, converting `OrderEvent` to `FillEvent` at $t+1$ bar open price.
  - `portfolio.py`: Portfolio manager tracking cash, positions, margin, and mark-to-market NAV.
  - `engine.py`: Master discrete event loop executing causal bar steps.
- `execution/fill_models.py` & `slippage_models/`:
  - `LOBSimulator`: Level-2 limit order book queue position tracking with cancellation decay.
  - Kyle's Lambda square-root price impact and Hasbrouck slippage models.
  - Tiered exchange commission models (Maker/Taker fees, SEC exchange transaction fees).
- `portfolio/`:
  - `markowitz/`: Modern Portfolio Theory mean-variance optimization and efficient frontier generation.
  - `black_litterman/`: Bayesian blending of market equilibrium prior returns with subjective strategy views.
  - `risk_parity/`: Hierarchical Risk Parity (HRP) dendrogram clustering and Equal Risk Contribution (ERC).
- `metrics/`:
  - `risk_measures/`: Annualized return, Sharpe, Sortino, Calmar, Cornish-Fisher Value-at-Risk (VaR), and Conditional Value-at-Risk (CVaR).
  - `drawdown_analysis/`: Underwater drawdown curves, maximum drawdown duration, recovery factors.
  - `overfitting/`: Deflated Sharpe Ratio (DSR) and CSCV Probability of Backtest Overfitting (PBO) solvers.
- `reports/tearsheet/`:
  - `app.py`: Full interactive Streamlit web dashboard.
  - `figures.py`: Plotly visualization renderers for calibration curves, equity curves, underwater waterfalls, and factor radars.
  - `stress.py`: Macroeconomic crisis replay and reverse stress testing solvers.

### 5.7 Alpha Signal Generation, Regime Filters & Arbitrage (`signals/`)
- `generation/`:
  - `crossover/`: Dual EMA and Golden Cross trend signals.
  - `breakout/`: Donchian channel and volatility breakout generators.
  - `mean_reversion/`: Extreme Bollinger Band and RSI mean-reversion signals.
- `filtering/`:
  - `volatility_filter/`: Regime filters blocking trade admissions during extreme GARCH volatility spikes.
  - `causal_gate/`: Filters blocking features that fail Granger causality and Transfer Entropy significance tests.
- `ranking/cross_sectional/`: Cross-sectional momentum decile rankers for long-short basket selection.
- `arbitrage/pairs_trading/`: Cointegrated pairs trading engine utilizing dynamic Kalman filter hedge ratios and rolling Z-score spread entries/exits.
- `persist.py`: Persists generated signals to MySQL/SQLite `trade_signals` table.

### 5.8 Orchestration Pipelines & Drift Monitoring (`pipelines/`)
- `etl/`: Automated pipeline orchestrating ingestion, cleaning, feature generation, and Parquet caching.
- `training/`: Purged Walk-Forward Cross-Validation training pipelines with combinatorial embargoing.
- `inference/daily.py`: Daily batch inference loop loading fresh bars, generating point-in-time features, evaluating ML models, and outputting trade signals.
- `monitoring/drift.py`: Statistical concept drift monitor calculating two-sample Kolmogorov-Smirnov test statistics and Population Stability Index (PSI) to detect feature distribution shifts.
- `run_demo.py`: Master end-to-end demo orchestrator running synthetic data generation, feature engineering, ML training, event-driven backtesting, database persistence, and report generation in a single unified command.

### 5.9 Research Protocols, Ablation & Microstructure Calibration (`research/`)
- `protocol.py`: Pre-registered research protocol defining candidate strategy families, parameter grids, and objective validation survival rules.
- `experiment.py`: Immutable `ExperimentSpec` container with role-excluded canonical hashing, duplicate execution caching, and trial counting.
- `grid.py`: Deterministic parameter grid generator covering 55 parameter combinations across 6 strategy families.
- `selection.py`: Objective candidate selection choosing the best dev candidate per family under protocol constraints (trade count $\ge 30$, max drawdown $\ge -40\%$, highest net Sharpe) and evaluating validation survival.
- `ablation.py`: Architectural pillar decomposition (V0: raw signal, V1: +costs, V2: +vol regime, V3: +causal gate, V4: +meta-labeling, V5: full system) with paired block-bootstrap difference confidence intervals.
- `sensitivity.py`: Parameter sensitivity surfaces, plateau scores, and cost multiplier sweeps (0x to 4x) solving for break-even transaction costs.
- `real_stress.py`: Empirical historical stress calibration calibrating realized equity drawdowns for 2008 Lehman (-46.04%), 2020 COVID (-33.72%), and 2022 rate hikes (-24.50%).
- `microstructure/recorder.py`: High-frequency Level-2 order book depth and trade recorder connecting to live Binance REST/WebSocket endpoints with zero trade gaps and midpoint timestamping.
- `microstructure/calibrate.py`: Calibrates empirical spread, visible depth, order book imbalance IC, trade sign autocorrelation, and power-law market impact exponent ($\gamma=1.1988$).
- `forward.py`: Operational forward paper testing harness enforcing pre-registration time-barriers (signals emitted strictly before bar open).
- `lockbox.py` & `run_lockbox.py`: Cryptographically sealed out-of-sample lockbox vault CLI refusing execution without exact confirmation passphrase.
- `rebuild.py` & `reproduce.py`: 100% deterministic offline reproduction scripts rebuilding datasets from raw content-addressed archives and reproducing baseline, trial, and validation backtest metrics to $10^{-9}$ tolerance.

### 5.10 Cryptographic Integrity & Automated Verification Gate (`integrity/`)
- `ledger.py`: Tamper-evident SHA-256 append-only cryptographic ledger tracking every dataset, experiment, calibration, and test proof.
- `anchor.py`: Verifies ledger hash chains and appends cryptographic anchors to `research/ledger/ANCHORS.txt`.
- `provenance.py`: Resolves data provenance and content-addressed raw archive manifests.
- `record_proof.py`: Appends verified stage test proofs directly to the ledger.
- `render.py`: Ledger-backed template renderer and documentation anti-forgery linter ensuring zero hardcoded numbers near metric words outside GENERATED blocks.
- `verify_all.py`: Master cryptographic verification gate executing 10 automated verification stages (`--lock-sha` validation).

### 5.11 Mathematical Utilities & Core Helpers (`utils/`)
- `math/linalg.py`: Linear algebra helpers, matrix inversion safeguards, and Ledoit-Wolf covariance shrinkage.
- `dates/trading_dates.py`: Trading calendar conversions, holiday detection, and business day arithmetic.
- `profiling/profilers.py`: Execution time measurement decorators and memory footprint analyzers.
- `logging/logger.py`: Thread-safe structured console and file logger.

---

## 6. Verification, Hardening & Quality Invariants

The integrity of Tapewright-Quant is enforced through multiple automated verification layers:

### 1. Test Suite Mutation Hardening (`scripts/mutation_test.py`)
To ensure test suites genuinely catch subtle bugs rather than performing superficial assertions, an AST-based mutation testing harness injects deliberate code faults:
- **Arithmetic Mutator**: Swaps operators (`+` $\leftrightarrow$ `-`, `*` $\leftrightarrow$ `/`).
- **Comparison Mutator**: Inverts comparison operators (`<` $\leftrightarrow$ `<=`, `>` $\leftrightarrow$ `>=`).
- **Logical Mutator**: Inverts boolean logic (`and` $\leftrightarrow$ `or`).
- **Lookahead Off-by-One Mutator**: Shifts lag indexing (`t` $\leftrightarrow$ `t+1`, `shift(1)` $\leftrightarrow$ `shift(0)`).
- **Constant Mutator**: Modifies numeric constants (`0.0` $\to$ `1.0`, `1.0` $\to$ `0.0`).

**Mutation Kill Rates Achieved**:
- **Event Loop Engine**: **84.0% kill rate**
- **Execution Models**: **100.0% kill rate**
- **Risk Measures & Anti-Overfit**: **80.0% kill rate**
- **Fractional Differentiation**: **100.0% kill rate**
- **Feature Registry**: **100.0% kill rate**
- **Training Pipelines**: **100.0% kill rate**
- **Signal Generation**: **80.0% kill rate**

### 2. Metamorphic & Differential Invariant Testing (`tests/integration/test_metamorphic.py`)
Verifies non-negotiable physical and financial invariants:
- **Scale Invariance**: Multiplying initial portfolio cash by scalar $k > 0$ yields identical percentage returns, Sharpe ratios, and drawdowns.
- **Time-Reversal Sign Invariance**: Inverting price series direction inverts directional return signs.
- **Cash Conservation Invariant**: $\text{Cash Balance} + \sum (\text{Asset Positions} \times \text{Current Price}) = \text{Total NAV}$ verified to $< 10^{-6}$ numerical precision at every bar step.
- **Cost Monotonicity**: Increasing simulated transaction fees or slippage coefficients strictly decreases or maintains net portfolio return ($\frac{\partial \text{Return}}{\partial \text{Cost}} \le 0$).
- **Symbol Permutation Invariance**: Permuting the iteration order of asset symbols in multi-asset universes produces identical fills and NAV trajectories.
- **Lookahead Mutant Detection**: Invariant tests catch intentional future-peeking mutants, causing divergence $> 1326$ standard deviations.

### 3. The 10-Stage Verification Gate (`integrity/verify_all.py`)
Run via:
```powershell
python -m integrity.verify_all --lock-sha <LOCK_FILE_SHA256>
```
Executes all 10 verification stages sequentially:
1. `stage_lock`: Verifies test lock hash matches current suite checksum.
2. `stage_ledger_chain`: Verifies SHA-256 hash chain validity across all ledger blocks.
3. `stage_constraints`: Runs `scripts/check_constraints.py` (0 violations of Python/MySQL-only rules).
4. `stage_test_integrity`: Runs `scripts/check_test_integrity.py` (enforces rules R1 through R7).
5. `stage_claims_audit`: Audits empirical findings against pre-registered claims in `claims.json`.
6. `stage_docs_lint`: Audits all `.md` files for hardcoded metrics outside GENERATED blocks.
7. `stage_pytest`: Executes full test suite (333 passed, 1 skipped cleanly for external MySQL).
8. `stage_forward`: Verifies forward test emission time barriers and outcome causality.
9. `stage_reproduce`: Performs offline dataset rebuild and verifies deterministic backtest reproduction.
10. `stage_research`: Verifies research discipline, trial counts, and lockbox isolation status.

---

## 7. Empirical Research Findings & The Scientific Truth

Unlike marketing claims of 900% returns, Tapewright-Quant documents real empirical outcomes on real market data (Liquid US ETFs 2005–2022 and Binance crypto pairs 2017–2023) directly from the cryptographically hashed ledger:

### Real-Data Strategy Performance Across Development & Validation Splits
Evaluated across 6 strategy families on the historical development split (`dev`: 2005–2019 for equities, 2017–2021 for crypto) and audited on the held-out validation split (`val`: 2020–2022 for equities, 2022–2023 for crypto):

| Strategy Family | Dev Net Sharpe | Dev Ann Return | Dev Max Drawdown | Val Net Sharpe | Val PSR (0) | Dev DSR | Survives Protocol? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Breakout** | -0.4273 | -1.27% | -26.22% | -0.9842 | 0.0159 | 0.0000 | **False** |
| **Crossover** | 0.6457 | +8.41% | -32.48% | 0.3059 | 0.7505 | 0.0360 | **False** |
| **Mean Reversion** | 0.2169 | +0.17% | -1.90% | -0.4937 | 0.1427 | 0.0000 | **False** |
| **ML Meta** | 0.4913 | +6.19% | -39.48% | 0.4258 | 0.8268 | 0.0072 | **False** |
| **Pairs Arbitrage** | 0.5197 | +6.04% | -36.84% | 0.4717 | 0.8526 | 0.0090 | **False** |
| **Cross-Sectional Momentum** | 0.6226 | +8.33% | -31.90% | 0.8222 | 0.9652 | 0.0284 | **False** |

### The Honest Empirical Outcome
Under the pre-registered protocol survival rule:
1. Development Deflated Sharpe Ratio (DSR) $\ge 0.90$
2. Validation Net Sharpe $> 0.0$
3. Validation Probabilistic Sharpe Ratio against Zero (PSR) $\ge 0.90$

**Zero candidates survived the pre-registered protocol rule.**
While strategies like Cross-Sectional Momentum achieved positive validation Sharpe (0.8222) and validation PSR (0.9652), the rigorous penalty for evaluating 55 parameter trials depressed its Deflated Sharpe Ratio (DSR = 0.0284) below the 0.90 threshold.

**Why this is the greatest portfolio demonstration possible**:
Amateur developers fake results or lower thresholds when their models don't pass. Tapewright-Quant documents this honest scientific outcome without relaxing thresholds or cherry-picking. This demonstrates to institutional quant recruiters that the engineer understands:
- What selection bias and the multiple testing problem actually do to Sharpe ratios.
- That genuine market alpha is extremely scarce and cannot be discovered by basic parameter sweeps.
- How to build a research system that prevents self-deception and data snooping.

---

## 8. Comprehensive Directory & File Map

```
Tapewright-quant/
├── .env.example                               # Environment variable template
├── .gitignore                                 # Git ignore patterns
├── CLAUDE_ARCHITECTURAL_BLUEPRINT.md          # 5-Pillar Architectural upgrade blueprint
├── PROGRESS.md                                # Complete 23-prompt chronological execution log
├── PROJECT_SPEC.md                            # Original master specification and constraints
├── README.md                                  # Ledger-generated master documentation
├── SYSTEM_STATUS.md                           # System implementation status and invariant audits
├── pytest.ini                                 # Pytest configuration
├── requirements.txt                           # Strict minimal dependencies
│
├── backtesting/                               # DISCRETE-EVENT BACKTESTING ENGINE
│   ├── engine/
│   │   └── event_loop/                        # Event loop, data handlers, execution, portfolio
│   ├── execution/                             # LOB queue simulator, Kyle lambda, slippage models
│   ├── metrics/                               # Risk measures, drawdowns, DSR, CSCV PBO solvers
│   ├── portfolio/                             # Markowitz, Black-Litterman, HRP, ERC optimizers
│   └── reports/tearsheet/                     # Interactive Streamlit app, Plotly renderers, stress replay
│
├── config/                                    # SYSTEM CONFIGURATION
│   ├── database/                              # MySQL & SQLite connection parameters
│   ├── environments/                          # Environment variables and run mode loaders
│   └── trading/                               # Asset universe, trading calendars, risk limits
│
├── data/                                      # DATA INGESTION & STORAGE
│   ├── caching/parquet_store.py               # Partitioned columnar Parquet store (symbol/year)
│   ├── cleaning/                              # MAD outlier filters, corporate actions, split adjustments
│   ├── ingestion/                             # Equities, CCXT crypto, FX, FRED rates, SEC EDGAR, synthetic
│   ├── transformation/                        # Dollar/volume bar aggregators and resamplers
│   └── validation/                            # Schema rule validators
│
├── database/                                  # RELATIONAL PERSISTENCE (MINIMAL-DB)
│   ├── connection/                            # Thread-safe connection pool manager (MySQL/SQLite)
│   ├── migrations/                            # Pure Python database schema migration runners
│   ├── repositories/                          # Repositories: assets, market_data, signals, orders, portfolio
│   └── schema/schema.sql                      # Production MySQL DDL (6 core tables)
│
├── docs/                                      # LEDGER-RENDERED RESEARCH DOCUMENTATION
│   ├── FINDINGS.md                            # Empirical research findings & validation results
│   ├── FORWARD_TEST.md                        # Forward paper testing operational guide
│   ├── LIMITATIONS.md                         # Transparent record of protocol biases & assumptions
│   ├── METHODOLOGY.md                         # Mathematical derivations & causal gating specs
│   └── templates/                             # Ledger-backed .md.tpl documentation templates
│
├── features/                                  # QUANTITATIVE FEATURE LIBRARY
│   ├── econometric/                           # Fractional diff (FFD), cointegration, Kalman filter, causal graph
│   ├── market_microstructure/                 # Roll/Corwin-Schultz spreads, order book imbalance, Shannon entropy, VPIN
│   ├── sentiment/                             # Loughran-McDonald lexicon scorer, transformer text embeddings
│   ├── store/registry/                        # Point-in-time feature catalog (preventing lookahead bias)
│   └── technical_indicators/                  # Momentum, Volatility, Trend, Volume indicators
│
├── integrity/                                 # TRUST LAYER & CRYPTOGRAPHIC VERIFICATION GATE
│   ├── anchor.py                              # Ledger hash chain verification & external time anchoring
│   ├── ledger.py                              # Append-only SHA-256 cryptographic audit ledger
│   ├── provenance.py                          # Raw archive content-addressed provenance
│   ├── record_proof.py                        # Automated proof recorder
│   ├── render.py                              # Template renderer & anti-forgery documentation linter
│   ├── stages/                                # 10 verification stage runners
│   └── verify_all.py                          # Master verification gate CLI
│
├── models/                                    # PREDICTIVE & MACHINE LEARNING MODELS
│   ├── deep_learning/                         # PyTorch SimpleTFT (VSN + attention), WaveNet, LSTM
│   ├── ensemble/                              # XGBoost, LightGBM, CatBoost with Purged CV & permutation importance
│   ├── registry.py                            # Model hyperparameter and metric registry
│   ├── reinforcement_learning/                # Gymnasium TradingEnv, Double-DQN, PPO, Differential Sharpe
│   └── time_series/                           # Auto-ARIMA, GARCH(1,1), AdditiveTrendSeasonal, StateSpace
│
├── pipelines/                                 # ORCHESTRATION PIPELINES
│   ├── etl/                                   # Ingestion-to-feature store pipeline
│   ├── inference/daily.py                     # Daily point-in-time inference batch pipeline
│   ├── monitoring/drift.py                    # Kolmogorov-Smirnov & PSI statistical drift monitor
│   ├── run_demo.py                            # Master end-to-end demo runner
│   └── training/                              # Purged walk-forward model training orchestrator
│
├── research/                                  # PROTOCOLS, ABLATION & EMPIRICAL RESEARCH
│   ├── ablation.py                            # V0-V5 pillar decomposition & paired block-bootstrap CIs
│   ├── calibration/params.json                # Empirical Binance L2 calibrated parameters
│   ├── experiment.py                          # Immutable ExperimentSpec & duplicate caching
│   ├── forward.py                             # Forward paper testing harness & time barrier enforcement
│   ├── grid.py                                # Deterministic parameter grid generator (55 trials)
│   ├── ledger/                                # Cryptographic ledger JSONL & ANCHORS.txt
│   ├── lockbox.py                             # Sealed out-of-sample lockbox vault isolation
│   ├── microstructure/                        # Real Binance L2 depth/trade recorder & calibrator
│   ├── protocol.py                            # Pre-registered protocol specifications
│   ├── raw_archive/                           # Content-addressed raw dataset archive
│   ├── real_data.py                           # Real dataset loaders (ETFs & Binance pairs)
│   ├── real_stress.py                         # Calibrated historical crisis factor replay
│   ├── rebuild.py                             # 100% offline dataset rebuild
│   ├── reproduce.py                           # 100% offline deterministic experiment reproduction
│   ├── run_dev.py / run_val.py                # Dev and Val experiment pipeline CLIs
│   ├── selection.py                           # Objective candidate selection & protocol survival rule
│   ├── sensitivity.py                         # Parameter sensitivity surfaces & cost break-even sweeps
│   └── vault/                                 # Cryptographically sealed lockbox data (UNOPENED)
│
├── scripts/                                   # CODE QUALITY & HARNESS TOOLS
│   ├── check_constraints.py                   # Enforces Python/MySQL-only and non-empty function rules
│   ├── check_syntax.py                        # Compiles all AST trees to verify 0 syntax errors
│   ├── check_test_integrity.py                # Enforces integrity rules R1 through R7
│   └── mutation_test.py                       # AST mutation testing harness (operator/logic/lookahead mutants)
│
├── signals/                                   # SIGNAL GENERATION & ARBITRAGE
│   ├── arbitrage/pairs_trading/               # Cointegrated pairs trading with dynamic Kalman hedge ratios
│   ├── filtering/                             # GARCH volatility regime filter & causal graph gates
│   ├── generation/                            # Crossover, Breakout, Mean Reversion signals
│   ├── persist.py                             # Signal persistence to relational DB
│   └── ranking/cross_sectional/               # Cross-sectional momentum decile rankers
│
├── tests/                                     # COMPREHENSIVE TEST SUITE (333+ TESTS)
│   ├── SKIP_WHITELIST.txt                     # Whitelist for authorized external test skips
│   ├── integration/                           # Metamorphic tests, zero-lookahead audits, DB parity
│   └── unit/                                  # Unit tests covering every single module and math equation
│
└── utils/                                     # MATHEMATICAL & SYSTEM UTILITIES
    ├── dates/trading_dates.py                 # Trading calendars and business day arithmetic
    ├── logging/logger.py                      # Thread-safe structured logging
    ├── math/linalg.py                         # Ledoit-Wolf shrinkage, matrix inversion safeguards
    └── profiling/profilers.py                 # Execution time & memory profiling decorators
```

---

## 9. Briefing Prompt for ChatGPT

When you feed this document into ChatGPT or any advanced AI reasoning model, prefix it with the following prompt to give it instant mastery over the codebase:

```text
You are an elite Lead Quantitative Portfolio Manager and Algorithmic Trading Systems Architect at a top-tier quantitative hedge fund (Two Sigma / Citadel / Renaissance Technologies / AQR).

I am providing you with the complete architectural specification, mathematical foundation, and implementation guide for "Tapewright-Quant" (research identity: FinPulse-Apex), a production-grade, strictly causal, microstructure-realistic quantitative trading and research engine.

Review this document carefully. Understand:
1. How its 5 Institutional Pillars (Synthetic L2 Queue Matching with Kyle's Lambda, Marcos López de Prado Econometrics with Fractional Differentiation and Purged CV, Granger/Transfer Entropy Causal Discovery, Half-Kelly/HRP Risk Budgeting, and Calibrated Macro Crisis Stress-Testing) fundamentally differentiate it from toy GitHub trading bots.
2. The role of its Cryptographic Integrity Layer: append-only SHA-256 ledger, 10-stage automated verification gate, AST mutation testing (>80-100% kill rate), metamorphic invariant testing, offline deterministic reproducibility, and sealed lockbox holdout vault.
3. Its brutal scientific honesty: why reporting that 0 candidate models survived the pre-registered protocol rule (due to DSR trial penalties across 55 trials) demonstrates genuine institutional rigor over data mining.
4. Its complete data flow and modular codebase structure across config, data, database, features, models, backtesting, signals, pipelines, research, integrity, and tests.

Answer all subsequent questions about this repository with deep mathematical and institutional precision based on this specification.
```
