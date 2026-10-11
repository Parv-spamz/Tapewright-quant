<!-- GENERATED from ledger head 6640eae2b165e9482ae8d7cd2196e3ed19f132659b8e4baa8fe3b94788541f4c. Do not edit by hand. -->
<!-- GENERATED:BEGIN -->
# FinPulse-Apex: Research & Backtesting Methodology

## 1. Strictly Causal Discrete-Event Architecture
FinPulse-Apex enforces temporal causality via an event-driven discrete simulation engine. Every calculation is indexed strictly to historical bars available at bar close time $t$. Orders fill at the open of bar $t+1$, preventing same-bar lookahead and retroactive execution bias.

### Core Simulation Loop
```
[Market Data Ingestion] -> [HistoricBarDataHandler] -> [TickEvent (time t)]
                                 |
                                 v
[Strategy Logic] -------> [SignalEvent (target weight)]
                                 |
                                 v
[Portfolio Sizing] -----> [OrderEvent (shares, direction)]
                                 |
                                 v
[Execution Handler] ----> [FillEvent (fill_price, costs, time t+1 open)]
                                 |
                                 v
[Portfolio Accounting] -> [Equity & NAV State Update]
```

## 2. Microstructure-Realistic Execution Mechanics
Simulated fills incorporate empirical market friction components rather than zero-cost assumptions:
- **Exchange Fees & Commissions**: Fixed ticket fees plus proportional basis points reflecting institutional schedules.
- **Kyle's Lambda & Square-Root Market Impact**: Price impact scales proportionally to order size relative to average daily volume and volatility:
  $$\Delta P = c \cdot \sigma \cdot \left(\frac{Q}{ADV}\right)^\gamma$$
- **Level-2 Limit Order Book Simulation**: Five visible order book price tiers with queue position estimation and cancellation decay modeling.

## 3. Causal Feature Selection & Directed Information Flow
To avoid spurious correlations and multicollinearity, candidate technical features pass through predictive causality gates:
- **Granger Causality**: Evaluates whether past values of feature $X$ contain statistically significant information for predicting midprice returns $Y$ beyond past values of $Y$ alone.
- **Transfer Entropy**: Non-parametric information theoretic metric quantifying directed information transfer between non-linear processes without Gaussian assumptions.
- **False Discovery Rate (FDR) Control**: Simultaneous hypothesis testing across $N$ candidate predictors is controlled via the Benjamini-Hochberg step-up procedure.

## 4. Lopez de Prado Machine Learning & Labeling Framework
- **Triple Barrier Labeling**: Labels observations based on the first barrier touched: horizontal profit-taking, horizontal stop-loss, or vertical time horizon.
- **Purged Cross-Validation**: Standard k-fold cross-validation is fundamentally flawed on financial time series due to label overlap across fold boundaries. `PurgedKFold` purges training observations whose label intervals $[t_0, t_1]$ overlap with test intervals.
- **Embargoing**: Discards training observations immediately following test sets to prevent auto-regressive information leakage.
- **Meta-Labeling**: Secondary classification layer filtering false-positive trade signals produced by primary directional heuristics.

## 5. Statistical Anti-Overfitting Safeguards
- **Deflated Sharpe Ratio (DSR)**: Corrects the observed Sharpe ratio for multi-testing selection bias, non-normal returns (skewness and kurtosis), and sample length:
  $$DSR = \Phi\left(\frac{(SR - SR_0)\sqrt{T-1}}{\sqrt{1 - \gamma_3 SR + \frac{\gamma_4 - 1}{4} SR^2}}\right)$$
  where $SR_0$ is the expected maximum Sharpe ratio under the null hypothesis across $N$ evaluated trials.
- **Combinatorially Symmetric Cross-Validation (CSCV)**: Partitions $T$ historical observations into $S$ slices, forming all combinations $C(S, S/2)$ of training and test sets to compute the empirical Probability of Backtest Overfitting (PBO).
- **One-Shot Lockbox Holdout**: High-integrity out-of-sample holdout dataset sealed in cryptographic vault storage and programmatically isolated from all parameter tuning and model selection routines.
<!-- GENERATED:END -->
