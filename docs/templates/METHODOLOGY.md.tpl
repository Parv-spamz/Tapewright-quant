# Engine Methodology & Mathematics

## 1. Strictly Causal Discrete-Event Architecture

The platform enforces point-in-time causality through a discrete-event simulation loop. Market data, signal generation, portfolio sizing, and order matching are separated by explicit temporal boundaries.

### Event Processing Order

```
[HistoricBarDataHandler] ──► TickEvent (Bar t Close)
                                    │
                                    ▼
[Strategy / Alpha Model] ──► SignalEvent (Target Weights)
                                    │
                                    ▼
[Portfolio Sizer / Risk] ──► OrderEvent (Share Delta)
                                    │
                                    ▼
[Execution Simulator]    ──► FillEvent (Bar t+1 Open Price)
                                    │
                                    ▼
[Accounting Manager]     ──► Mark-to-Market State Update
```

### Temporal Execution Barrier

$$\mathcal{F}_t = \sigma(X_\tau : \tau \le t)$$

$$O_t = f(\mathcal{F}_t)$$

$$P_{\text{fill}, t+1} = P_{\text{open}, t+1} + \operatorname{Slippage}(O_t) + \Delta P_{\text{impact}, t+1}$$

Where:
* $\mathcal{F}_t$: Information filtration containing observations up to bar close $t$.
* $O_t$: Target order generated at bar close $t$.
* $P_{\text{fill}, t+1}$: Effective execution price executed at bar $t+1$ open.
* $P_{\text{open}, t+1}$: Market open price of bar $t+1$.
* $\Delta P_{\text{impact}, t+1}$: Endogenous market impact displacement.

## 2. Fractional Differentiation (FFD)

Standard integer differencing ($d=1$) removes non-stationarity at the cost of erasing long-memory cointegration signals. Fractional differentiation preserves memory while achieving stationarity.

### Binomial Expansion

$$(1-B)^d = \sum_{k=0}^{\infty} (-1)^k \binom{d}{k} B^k = \sum_{k=0}^{\infty} w_k B^k$$

Where weights $w_k$ are recursively computed:

$$w_k = -w_{k-1} \cdot \frac{d - k + 1}{k}, \quad w_0 = 1$$

Where:
* $B$: Backshift lag operator where $B^k X_t = X_{t-k}$.
* $d$: Real-valued fractional differencing order ($0 \le d \le 1$).
* $w_k$: Weight coefficient assigned to historical lag $k$.
* $\tau$: Minimum weight cutoff threshold ($|w_k| \ge \tau$) truncating the memory window.

## 3. Level-2 Limit Order Book & Market Impact

Execution models account for queue dynamics, adverse selection, and non-linear market impact.

### Level-2 Queue Priority & Depletion

$$Q_{t+1}^{\text{ahead}} = \max\left(0, Q_t^{\text{ahead}} - V_{t+1}^{\text{trade}}\right)$$

Where:
* $Q_t^{\text{ahead}}$: Unfilled quantity positioned ahead in the price tier queue.
* $V_{t+1}^{\text{trade}}$: Executed market volume at the matching price tier during interval $t+1$.

### Kyle's Square-Root Market Impact

$$\Delta P_t = c \cdot \sigma_t \cdot \left(\frac{V_t}{\text{ADV}_t}\right)^\gamma \cdot \operatorname{sign}(Q_t)$$

Where:
* $\Delta P_t$: Price displacement caused by order execution.
* $c$: Calibrated institutional impact scaling constant.
* $\sigma_t$: Historical daily price return volatility.
* $V_t$: Executed order volume in shares.
* $\text{ADV}_t$: 30-day average daily volume.
* $\gamma$: Concave elasticity exponent calibrated to empirical limit books.
* $\operatorname{sign}(Q_t)$: Order side indicator ($+1$ for buy, $-1$ for sell).

### Roll Effective Spread

$$S_{\text{Roll}} = 2 \sqrt{-\operatorname{Cov}(\Delta P_t, \Delta P_{t-1})}$$

Where:
* $S_{\text{Roll}}$: Inferred effective bid-ask spread.
* $\Delta P_t$: Price difference between consecutive transactions $P_t - P_{t-1}$.

## 4. Predictive Causal Graph Discovery

Spurious correlations are eliminated prior to model training through directed causal discovery and false discovery rate control.

### Bivariate Granger Causality

Unrestricted autoregressive model:

$$Y_t = \sum_{i=1}^{p} \alpha_i Y_{t-i} + \sum_{j=1}^{p} \beta_j X_{t-j} + \epsilon_t$$

Restricted autoregressive model:

$$Y_t = \sum_{i=1}^{p} \gamma_i Y_{t-i} + u_t$$

Wald $F$-statistic test:

$$F = \frac{(\text{RSS}_R - \text{RSS}_U) / p}{\text{RSS}_U / (T - 2p - 1)}$$

Where:
* $Y_t$: Target asset return process.
* $X_t$: Candidate predictor feature process.
* $p$: Optimal autoregressive lag order selected via Bayesian Information Criterion (BIC).
* $\text{RSS}_R$: Residual sum of squares from the restricted autoregression.
* $\text{RSS}_U$: Residual sum of squares from the unrestricted regression.
* $T$: Number of valid time-series observations.

### Non-Parametric Transfer Entropy

$$\text{TE}_{X \to Y} = \sum p(y_{t+1}, y_t^{(k)}, x_t^{(l)}) \log_2 \frac{p(y_{t+1} \mid y_t^{(k)}, x_t^{(l)})}{p(y_{t+1} \mid y_t^{(k)})}$$

Where:
* $\text{TE}_{X \to Y}$: Information flow from process $X$ to process $Y$.
* $y_t^{(k)}$: $k$-history trajectory of process $Y$.
* $x_t^{(l)}$: $l$-history trajectory of process $X$.

### Benjamini-Hochberg False Discovery Rate (FDR)

$$p_{(k)} \le \frac{k}{M} \cdot q$$

Where:
* $p_{(k)}$: Ordered $p$-values such that $p_{(1)} \le p_{(2)} \le \dots \le p_{(M)}$.
* $M$: Total number of simultaneous hypothesis tests.
* $q$: Target false discovery rate significance threshold ($q \le 0.05$).
* $k$: Highest rank index satisfying the step-up inequality.

## 5. Triple-Barrier Labeling & Meta-Labeling

Observations are labeled using dynamic horizontal profit/loss thresholds and vertical time horizons.

### Barrier Formulations

$$\tau_{\text{upper}} = P_{t_0} \cdot (1 + L \cdot \sigma_{t_0})$$

$$\tau_{\text{lower}} = P_{t_0} \cdot (1 - L \cdot \sigma_{t_0})$$

$$\tau_{\text{vertical}} = t_0 + H$$

Where:
* $P_{t_0}$: Asset midprice at event inception time $t_0$.
* $L$: Volatility multiplier scaling horizontal exit thresholds.
* $\sigma_{t_0}$: Exponentially weighted return volatility measured at time $t_0$.
* $H$: Maximum permissible holding period duration in bars.

### Meta-Labeling Binary Classification

$$y_i^* = \begin{cases} 1, & \text{if } y_i \cdot r_{i} > 0 \\ 0, & \text{otherwise} \end{cases}$$

Where:
* $y_i^*$: Binary meta-label predicting primary trade profitability.
* $y_i$: Directional trade signal emitted by primary model ($y_i \in \{-1, +1\}$).
* $r_i$: Realized return from trade entry to barrier intersection.

## 6. Purged Cross-Validation & Combinatorial Embargoing

Financial time series exhibit serial correlation and overlapping labels that cause standard k-fold cross-validation to leak future information.

### Purging Mechanics

A training observation $j$ is purged from the dataset if its event horizon overlaps with the evaluation interval of test fold $k$:

$$[t_{0,j}, t_{1,j}] \cap [T_0, T_1] \neq \emptyset$$

Where:
* $[t_{0,j}, t_{1,j}]$: Interval between feature timestamp $t_{0,j}$ and barrier touch timestamp $t_{1,j}$.
* $[T_0, T_1]$: Temporal boundary of the evaluation test fold.

### Embargo Mechanics

Autoregressive information leakage persists immediately after the test fold. Training observations following test fold $k$ are embargoed across duration $h$:

$$t_{0,j} \in [T_1, T_1 + h]$$

Where:
* $T_1$: End timestamp of the test fold.
* $h$: Embargo window duration derived from autoregressive decay length.

## 7. Hierarchical Risk Parity (HRP) & Equal Risk Contribution (ERC)

HRP constructs diversified portfolios without inverting ill-conditioned covariance matrices.

### Tree Distance Metric

$$d_{i,j} = \sqrt{\frac{1}{2} (1 - \rho_{i,j})}$$

Where:
* $d_{i,j}$: Normalized correlation distance metric between assets $i$ and $j$.
* $\rho_{i,j}$: Pearson correlation coefficient.

### Recursive Bisection Weight Allocation

$$\tilde{V}_k = w_k^T \Sigma_k w_k, \quad w_k = \frac{\operatorname{diag}(\Sigma_k)^{-1}}{\mathbf{1}^T \operatorname{diag}(\Sigma_k)^{-1} \mathbf{1}}$$

$$\alpha_1 = 1 - \frac{\tilde{V}_L}{\tilde{V}_L + \tilde{V}_R}, \quad \alpha_2 = 1 - \alpha_1$$

Where:
* $\Sigma_k$: Covariance sub-matrix of cluster $k$ shrunk via Ledoit-Wolf estimator.
* $\tilde{V}_L, \tilde{V}_R$: Variances of adjacent bisection sub-clusters $L$ and $R$.
* $\alpha_1, \alpha_2$: Proportional split allocation weights assigned to sub-clusters.

### Equal Risk Contribution (ERC)

$$RC_i = w_i \cdot \frac{(\Sigma w)_i}{\sqrt{w^T \Sigma w}} = \frac{1}{N} \sqrt{w^T \Sigma w}, \quad \forall i$$

Where:
* $RC_i$: Risk contribution of asset $i$ to total portfolio volatility.
* $w$: Asset weight allocation vector.
* $\Sigma$: Asset return covariance matrix.
* $N$: Total number of portfolio assets.

## 8. Statistical Anti-Overfitting Safeguards

### Deflated Sharpe Ratio (DSR)

$$\text{DSR} = \Phi\left( \frac{(\widehat{\text{SR}} - \text{SR}_0) \sqrt{T - 1}}{\sqrt{1 - \hat{\gamma}_3 \widehat{\text{SR}} + \frac{\hat{\gamma}_4 - 1}{4} \widehat{\text{SR}}^2}} \right)$$

$$\text{SR}_0 = \sqrt{2 \ln N} + \frac{\gamma_E}{\sqrt{2 \ln N}} - \frac{\ln(4\pi) + \ln(\ln N)}{2\sqrt{2 \ln N}}$$

Where:
* $\widehat{\text{SR}}$: Observed annualized Sharpe ratio.
* $\text{SR}_0$: Expected maximum Sharpe ratio under the null hypothesis across $N$ independent trials.
* $T$: Number of historical return observations.
* $\hat{\gamma}_3$: Empirical skewness of the return distribution.
* $\hat{\gamma}_4$: Empirical kurtosis of the return distribution.
* $N$: Total number of evaluated backtest parameter trials.
* $\gamma_E$: Euler-Mascheroni constant ($\approx 0.5772156649$).
* $\Phi$: Standard cumulative normal distribution function.

### Combinatorially Symmetric Cross-Validation (CSCV) & PBO

$$\text{PBO} = \frac{1}{K} \sum_{k=1}^{K} \mathbb{I}\left( \text{SR}_{c_k^*}^{\text{val}} < \operatorname{median}(\mathbf{SR}_k^{\text{val}}) \right)$$

Where:
* $K$: Total number of combinatorial splits $C(S, S/2)$ formed by CSCV.
* $c_k^*$: Candidate strategy optimizing in-sample performance on split $k$.
* $\text{SR}_{c_k^*}^{\text{val}}$: Out-of-sample validation Sharpe ratio of strategy $c_k^*$.
* $\mathbf{SR}_k^{\text{val}}$: Distribution of validation Sharpe ratios across all strategies in split $k$.
* $\mathbb{I}(\cdot)$: Indicator function equal to 1 if the condition holds, 0 otherwise.
