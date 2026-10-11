# Engine Methodology & Mathematics

## 1. Discrete-Event Temporal Causality

Orders emitted at bar $t$ close execute at bar $t+1$ open. The filtration is strictly point-in-time:

$$\mathcal{F}_t = \sigma(X_\tau : \tau \le t), \quad O_t = f(\mathcal{F}_t)$$

$$P_{\text{fill}, t+1} = P_{\text{open}, t+1} + \operatorname{Slippage}(O_t) + \Delta P_{\text{impact}, t+1}$$

Where:
* $\mathcal{F}_t$: Information filtration at bar close $t$.
* $O_t$: Order vector generated at bar close $t$.
* $P_{\text{fill}, t+1}$: Execution price at bar $t+1$ open.
* $P_{\text{open}, t+1}$: Market open price of bar $t+1$.
* $\Delta P_{\text{impact}, t+1}$: Endogenous market impact displacement.

## 2. Fractional Differentiation (FFD)

The fractional differentiation operator $(1-B)^d$ achieves stationarity while maximizing memory retention:

$$(1-B)^d = \sum_{k=0}^{\infty} w_k B^k, \quad w_k = -w_{k-1} \cdot \frac{d - k + 1}{k}, \quad w_0 = 1$$

Where:
* $B$: Backshift lag operator ($B^k X_t = X_{t-k}$).
* $d$: Real-valued fractional differencing order ($0 \le d \le 1$).
* $w_k$: Binomial series weight at lag $k$.

## 3. Limit Order Book & Market Impact

### Level-2 Queue Depletion

A limit order fills only after preceding queue volume clears:

$$Q_{t+1}^{\text{ahead}} = \max\left(0, Q_t^{\text{ahead}} - V_{t+1}^{\text{trade}}\right)$$

Where:
* $Q_t^{\text{ahead}}$: Queue volume ahead of order at time $t$.
* $V_{t+1}^{\text{trade}}$: Traded volume at price level during interval $t+1$.

### Kyle's Square-Root Market Impact

Price displacement scales non-linearly with order size relative to daily volume:

$$\Delta P_t = c \cdot \sigma_t \cdot \left(\frac{V_t}{\text{ADV}_t}\right)^\gamma \cdot \operatorname{sign}(Q_t)$$

Where:
* $\Delta P_t$: Execution price displacement.
* $c$: Calibrated scaling coefficient.
* $\sigma_t$: Daily return volatility.
* $V_t$: Order volume in shares.
* $\text{ADV}_t$: 30-day average daily volume.
* $\gamma$: Concave elasticity exponent.
* $\operatorname{sign}(Q_t)$: Direction ($+1$ for buy, $-1$ for sell).

### Roll Effective Spread

Estimates effective spread from trade return covariance:

$$S_{\text{Roll}} = 2 \sqrt{-\operatorname{Cov}(\Delta P_t, \Delta P_{t-1})}$$

Where:
* $S_{\text{Roll}}$: Effective bid-ask spread.
* $\Delta P_t$: Trade-to-trade price change $P_t - P_{t-1}$.

## 4. Predictive Causal Graph Discovery

### Bivariate Granger Causality

Tests if predictor series $X$ improves return forecast $Y$ beyond autoregressive lags:

$$F = \frac{(\text{RSS}_R - \text{RSS}_U) / p}{\text{RSS}_U / (T - 2p - 1)}$$

Where:
* $F$: Wald test statistic.
* $\text{RSS}_R$: Residual sum of squares of restricted model $Y_t = \sum \gamma_i Y_{t-i} + u_t$.
* $\text{RSS}_U$: Residual sum of squares of unrestricted model $Y_t = \sum \alpha_i Y_{t-i} + \sum \beta_j X_{t-j} + \epsilon_t$.
* $p$: Lag order selected by BIC.
* $T$: Number of observations.

### Transfer Entropy

Quantifies non-parametric directed information flow:

$$\text{TE}_{X \to Y} = \sum p(y_{t+1}, y_t^{(k)}, x_t^{(l)}) \log_2 \frac{p(y_{t+1} \mid y_t^{(k)}, x_t^{(l)})}{p(y_{t+1} \mid y_t^{(k)})}$$

Where:
* $\text{TE}_{X \to Y}$: Directed information from $X$ to $Y$.
* $y_t^{(k)}$: History state of process $Y$.
* $x_t^{(l)}$: History state of process $X$.

### Benjamini-Hochberg FDR Control

Controls false discoveries across multiple pairwise tests:

$$p_{(k)} \le \frac{k}{M} \cdot q$$

Where:
* $p_{(k)}$: Sorted $p$-values ($p_{(1)} \le \dots \le p_{(M)}$).
* $M$: Total hypothesis tests.
* $q$: False discovery rate threshold ($q \le 0.05$).
* $k$: Maximum rank satisfying the threshold.

## 5. Labeling Framework

### Triple-Barrier Method

Events are labeled by the first threshold intersected:

$$\tau_{\text{upper}} = P_{t_0} (1 + L \sigma_{t_0}), \quad \tau_{\text{lower}} = P_{t_0} (1 - L \sigma_{t_0}), \quad \tau_{\text{vertical}} = t_0 + H$$

Where:
* $P_{t_0}$: Entry price at time $t_0$.
* $L$: Volatility multiplier.
* $\sigma_{t_0}$: Realized volatility.
* $H$: Vertical holding horizon in bars.

### Meta-Labeling

A secondary binary classifier filters primary heuristic bets:

$$y_i^* = \mathbb{I}(y_i \cdot r_i > 0)$$

Where:
* $y_i^*$: Binary meta-label.
* $y_i$: Directional prediction ($y_i \in \{-1, +1\}$).
* $r_i$: Realized trade return.
* $\mathbb{I}$: Indicator function.

## 6. Purged Cross-Validation & Embargoing

### Purging

Training samples whose label windows overlap test fold intervals are deleted:

$$[t_{0,j}, t_{1,j}] \cap [T_0, T_1] \neq \emptyset$$

Where:
* $[t_{0,j}, t_{1,j}]$: Event window from feature timestamp to barrier touch.
* $[T_0, T_1]$: Test fold boundaries.

### Embargoing

Training samples immediately following the test fold are dropped:

$$t_{0,j} \in [T_1, T_1 + h]$$

Where:
* $T_1$: End of test fold.
* $h$: Autoregressive embargo window.

## 7. Hierarchical Risk Parity (HRP)

### Correlation Distance

Computes topological tree distance between assets:

$$d_{i,j} = \sqrt{\frac{1}{2} (1 - \rho_{i,j})}$$

Where:
* $d_{i,j}$: Distance metric for single linkage hierarchical clustering.
* $\rho_{i,j}$: Pearson correlation between assets $i$ and $j$.

### Recursive Bisection Sizing

Allocates capital across tree clusters via inverse variance:

$$\tilde{V}_k = w_k^T \Sigma_k w_k, \quad \alpha_1 = 1 - \frac{\tilde{V}_L}{\tilde{V}_L + \tilde{V}_R}, \quad \alpha_2 = 1 - \alpha_1$$

Where:
* $\Sigma_k$: Covariance matrix of cluster $k$ with Ledoit-Wolf shrinkage.
* $\tilde{V}_L, \tilde{V}_R$: Variances of sub-clusters $L$ and $R$.
* $\alpha_1, \alpha_2$: Split allocation factors.

### Equal Risk Contribution (ERC)

Equalizes asset contributions to total volatility:

$$RC_i = w_i \cdot \frac{(\Sigma w)_i}{\sqrt{w^T \Sigma w}} = \frac{1}{N} \sqrt{w^T \Sigma w}, \quad \forall i$$

Where:
* $RC_i$: Risk contribution of asset $i$.
* $w$: Asset weight vector.
* $\Sigma$: Covariance matrix.
* $N$: Number of assets.

## 8. Anti-Overfitting Statistics

### Deflated Sharpe Ratio (DSR)

Adjusts observed Sharpe ratio for multi-testing selection bias, skewness, and kurtosis:

$$\text{DSR} = \Phi\left( \frac{(\widehat{\text{SR}} - \text{SR}_0) \sqrt{T - 1}}{\sqrt{1 - \hat{\gamma}_3 \widehat{\text{SR}} + \frac{\hat{\gamma}_4 - 1}{4} \widehat{\text{SR}}^2}} \right)$$

$$\text{SR}_0 = \sqrt{2 \ln N} + \frac{\gamma_E}{\sqrt{2 \ln N}} - \frac{\ln(4\pi) + \ln(\ln N)}{2\sqrt{2 \ln N}}$$

Where:
* $\widehat{\text{SR}}$: Observed Sharpe ratio.
* $\text{SR}_0$: Expected maximum Sharpe ratio under the null hypothesis.
* $T$: Number of observations.
* $\hat{\gamma}_3, \hat{\gamma}_4$: Return skewness and kurtosis.
* $N$: Number of tested strategy trials.
* $\gamma_E$: Euler-Mascheroni constant ($\approx 0.5772$).
* $\Phi$: Standard normal cumulative distribution function.

### Probability of Backtest Overfitting (PBO)

Evaluates combinatorial cross-validation splits to measure rank degradation:

$$\text{PBO} = \frac{1}{K} \sum_{k=1}^{K} \mathbb{I}\left( \text{SR}_{c_k^*}^{\text{val}} < \operatorname{median}(\mathbf{SR}_k^{\text{val}}) \right)$$

Where:
* $K$: Total combinatorial splits $C(S, S/2)$ generated via CSCV.
* $c_k^*$: In-sample optimal strategy on split $k$.
* $\text{SR}_{c_k^*}^{\text{val}}$: Out-of-sample Sharpe ratio of strategy $c_k^*$.
* $\mathbf{SR}_k^{\text{val}}$: Distribution of validation Sharpe ratios in split $k$.
* $\mathbb{I}$: Indicator function.
