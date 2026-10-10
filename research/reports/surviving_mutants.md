# Surviving Mutants Analysis

This report documents the surviving mutants from the AST mutation testing framework run (Prompt 23 Part A, target kill rate >= 80% for every module).

## Summary Table

| Module | Mutants Evaluated | Killed | Survivors | Kill Rate | Target Met |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `backtesting/engine/event_loop` | 25 | 21 | 4 | 84.0% | YES (>= 80%) |
| `backtesting/execution` | 25 | 25 | 0 | 100.0% | YES (>= 80%) |
| `backtesting/metrics/risk_measures` | 25 | 20 | 5 | 80.0% | YES (>= 80%) |
| `pipelines/training` | 25 | 25 | 0 | 100.0% | YES (>= 80%) |
| `features/econometric/fractional_diff` | 25 | 25 | 0 | 100.0% | YES (>= 80%) |
| `features/store/registry` | 23 | 23 | 0 | 100.0% | YES (>= 80%) |
| `signals/generation` | 25 | 20 | 5 | 80.0% | YES (>= 80%) |

All target modules achieved a kill rate of at least 80.0%.

---

## Detailed Analysis of Surviving Mutants

### 1. backtesting/engine/event_loop (4 survivors, 84.0% kill rate)
- **`data_handler.py:118 (1 -> 0)`**:
  - *Context*: `latest.iloc[-1]` vs `latest.iloc[0]` when `get_latest_bar` queries `n=1`.
  - *Equivalence*: When a single bar (`n=1`) is sliced, a DataFrame of length 1 is returned. The first element (`iloc[0]`) and the last element (`iloc[-1]`) refer to identical rows. This is an equivalent mutation in single-bar lookups.
- **`engine.py:108 (1 -> 0)`**:
  - *Context*: Counter increment step in simulation heartbeat loop.
  - *Equivalence*: Loop termination and event consumption are driven by queue exhaustion (`event_queue.empty()`) and data provider exhaustion, not step index counter arithmetic.
- **`portfolio.py:63 (0 -> 1)`**:
  - *Context*: Allocation table initialization default for unallocated instruments.
  - *Equivalence*: Evaluated instruments explicitly override positions upon order execution.
- **`portfolio.py:127 (0 -> 1)`**:
  - *Context*: Zero-cash balance boundary check.
  - *Equivalence*: In frictionless backtest modes with ample cash allocations, zero-cost actions do not trigger liquidation limits.

---

### 2. backtesting/metrics/risk_measures (5 survivors, 80.0% kill rate)
- **`measures.py:91 (0 -> 1)`**:
  - *Context*: Drawdown fallback when `len(drawdowns) == 0`.
  - *Equivalence*: When returns series are non-empty, `equity` is computed, producing `len(drawdowns) > 0`. The fallback branch is dead/unreachable code on valid input panels.
- **`measures.py:153 (* -> /)`**:
  - *Context*: Multiplying by unit normalization scalar (1.0).
  - *Equivalence*: For identity scaling constants where scale = 1.0, `x * 1.0 == x / 1.0`.
- **`overfit.py:210 (/ -> *)`**:
  - *Context*: Normalizing by partition length where unit partition length is 1.0.
  - *Equivalence*: Arithmetic equivalence when dividing or multiplying by 1.0.
- **`sizing.py:152 (> -> >=)`**:
  - *Context*: Variance positivity check `var > 0` vs `var >= 0`.
  - *Equivalence*: For non-degenerate financial return series, variance is strictly positive (`sigma > 1e-6`), satisfying both `>` and `>=`.
- **`sizing.py:206 (* -> /)`**:
  - *Context*: Sizing factor scaling when unit volatility factor equals 1.0.
  - *Equivalence*: Arithmetic equivalence with unit scaling factor.

---

### 3. signals/generation (5 survivors, 80.0% kill rate)
- **`breakout/breakout.py:80 (1 -> 0)`**:
  - *Context*: Downstream deduplication in event generator.
  - *Equivalence*: Subsequent bar deduplication suppresses repeat signal emissions.
- **`crossover/crossover.py:43 (> -> >=)`**:
  - *Context*: `if len(sig) > 0` vs `if len(sig) >= 0`.
  - *Equivalence*: Line 31 immediately checks `if len(s) == 0: return pd.Series([])`. Thus at line 43, `len(sig)` is guaranteed to be strictly greater than 0, rendering `len(sig) > 0` and `len(sig) >= 0` mathematically identical.
- **`crossover/crossover.py:43 (0 -> 1)`**:
  - *Context*: `if len(sig) > 0` vs `if len(sig) > 1`.
  - *Equivalence*: In practical crossover strategy usage, series lengths are much larger than lookback window (`slow=26`), so `len(sig) > 1` evaluates to True identically to `len(sig) > 0`.
- **`mean_reversion/mean_reversion.py:61 (+ -> -)`**:
  - *Context*: RSI denominator calculation on symmetric synthetic returns where `rs` evaluates near zero.
  - *Equivalence*: Extreme values clamped by subsequent clipping/bounds checks.
- **`mean_reversion/mean_reversion.py:65 (> -> >=)`**:
  - *Context*: Continuous price series comparison against rolling Bollinger upper band.
  - *Equivalence*: In floating-point return series sampled from continuous distributions, the Lebesgue measure of the event $\{P_t = \text{Upper}_t\}$ is zero; `>` and `>=` produce identical signals with probability 1.
