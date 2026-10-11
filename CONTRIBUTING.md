# Contributing to Tapewright-Quant

Thank you for your interest in contributing to **Tapewright-Quant** (research identity: **FinPulse-Apex**).

Tapewright-Quant is an institutional-grade quantitative research, econometrics, and discrete-event algorithmic execution engine. To protect scientific reproducibility, strict engineering constraints, and cryptographic provenance, all contributions must adhere to the standards outlined below.

---

## 1. Strict Technical & Language Constraints (Non-Negotiable)

Contributions must strictly comply with the following architecture constraints:

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
- **What MySQL / SQLite is used for**:
  - Relational metadata registries (`assets`, `model_registry`).
  - Quantitative signal audit trails (`trade_signals`).
  - Order execution records (`order_executions`).
  - High-level portfolio snapshots (`portfolio_snapshots`).
- **What MySQL / SQLite is NOT used for**:
  - Do **NOT** store high-frequency tick data, gigabytes of OHLCV bars, or high-dimensional feature arrays in SQL tables.
  - Historical price arrays, order book snapshots, and computed features must be handled in memory or stored in partitioned columnar Parquet files via `data/caching/parquet_store.py`.

---

## 2. Directory Structure & Module Standards

Every directory in the repository is a decoupled, importable Python package containing an `__init__.py`:

| Directory | Scope & Purpose |
| :--- | :--- |
| `backtesting/` | Discrete-event simulation loop, Level-2 queue execution, Kyle's lambda slippage, portfolio optimizers, tearsheets. |
| `config/` | Environment variable loaders, universe definitions, trading calendars, and risk parameters. |
| `data/` | Ingestion feeds (Equities, CCXT crypto, FX, FRED, SEC), synthetic stochastic generators, MAD filters, Parquet store. |
| `database/` | Minimal-DB connection pooling, schema DDL, Python migrations, and repository abstractions. |
| `features/` | Technical indicators, market microstructure spreads & entropy, fractional differentiation (FFD), causal graphs. |
| `models/` | ARIMA, GARCH(1,1), Tabular Ensembles (XGBoost, LightGBM, CatBoost), PyTorch sequence models, RL agents. |
| `signals/` | Signal generators (crossover, breakout, mean reversion), volatility regime filters, cross-sectional ranking. |
| `pipelines/` | ETL orchestration, daily inference batch loop, Kolmogorov-Smirnov drift monitoring, master run demo. |
| `research/` | Research protocols, 55-trial grid generator, candidate selection, ablations, Binance L2 calibration, lockbox. |
| `integrity/` | Trust layer: SHA-256 audit ledger, template renderer, anti-forgery linter, and 10-stage verification gate. |
| `scripts/` | Verification tools: syntax compiler, constraints checker, test-integrity checker, AST mutation test runner. |
| `tests/` | 333+ unit, integration, and metamorphic tests enforcing mathematical invariants and zero-lookahead safety. |

---

## 3. Test Suite Integrity Rules (R1–R7)

All tests are verified by `scripts/check_test_integrity.py` against strict rules:

- **R1 (Real Assertions)**: Every test function must contain at least one meaningful assertion or `pytest.raises` / `pytest.warns` context.
- **R2 (No Trivial Asserts)**: No `assert True`, `assert 1`, assert on constants, or self-comparisons (`assert x == x`).
- **R3 (Whitelist-Only Skips)**: No `pytest.skip`, `skipif`, `xfail`, or `importorskip` unless explicitly authorized in `tests/SKIP_WHITELIST.txt`.
- **R4 (No Mock Leaks)**: In production code (outside `tests/`), no `unittest.mock`, `MagicMock`, `monkeypatch`, or test-runner globals.
- **R5 (No Silenced Exceptions)**: In production code, no `except` handler that silently ignores errors (`pass` or `continue`) without re-raising or logging at `WARNING` or higher.
- **R6 (No Hardcoded Metrics in UI)**: In tearsheet visualization (`app.py`, `figures.py`), metrics must be dynamically computed or loaded from data—never hardcoded.
- **R7 (Parquet Storage Rules)**: Under `research/`, only designated data-access modules (`real_data.py`, `data_access.py`, `lockbox.py`) may instantiate `ParquetStore`.

---

## 4. Documentation & Anti-Forgery Policy

Documentation that reports quantitative metrics is generated dynamically from the cryptographic audit ledger:

1. **Templates First**: The canonical source for documentation containing research metrics is `docs/templates/*.md.tpl`. Never edit generated `.md` files directly.
2. **Dynamic Rendering**: Use the project's rendering engine:
   ```powershell
   python -c "from integrity.ledger import Ledger; from integrity.render import render_all; import shutil; l = Ledger('research/ledger/ledger.jsonl'); render_all(l); shutil.copyfile('docs/README.md', 'README.md')"
   ```
3. **Anti-Forgery Linter**: `integrity/render.py` enforces that no hardcoded numbers appear within 40 characters of metric keywords (such as `Sharpe`, `Drawdown`, `Return`, `Volatility`, `PBO`, `DSR`) outside authorized `<!-- GENERATED:BEGIN --> ... <!-- GENERATED:END -->` blocks or fenced code blocks.

---

## 5. Research Protocol & Lockbox Protection

- **Trial Counting Discipline**: Every parameter evaluation on the development dataset increments the trial counter in `research/ledger/` and is factored into the Deflated Sharpe Ratio (DSR) and CSCV Probability of Backtest Overfitting (PBO).
- **Sealed Lockbox Vault**: The held-out validation vault partition (`research/vault/`) is cryptographically sealed and programmatically guarded. **Never attempt to open or evaluate the lockbox**; access is reserved for single-use evaluations under verified protocol oversight.

---

## 6. Pre-Commit Verification Gate

Before submitting a pull request, run the local verification suite:

```powershell
# 1. Check syntax across all Python files
python scripts/check_syntax.py

# 2. Verify architectural and library constraints
python scripts/check_constraints.py

# 3. Verify test integrity rules (R1 through R7)
python scripts/check_test_integrity.py

# 4. Run documentation anti-forgery linter
python -c "from integrity.render import lint_docs; assert len(lint_docs()) == 0"

# 5. Run the unit and integration tests
pytest -v

# 6. Execute the full multi-stage integrity gate
python -m integrity.verify_all --lock-sha 8e7be5011149fd4f1e1fdbe64abd4cf5ae3a624e227ad7d5b251bc2f2548a45a --quick
```

All stages must report **PASS** with zero violations.

---

## 7. Pull Request Process

1. Create a dedicated feature branch from `main`:
   ```powershell
   git checkout -b feature/your-feature-name
   ```
2. Implement your changes, ensuring every new function includes real mathematical logic and corresponding unit tests.
3. Verify that no protected files (`tests/TEST_LOCK.json`, `research/vault/`, `research/ledger/`) were modified unintentionally.
4. Push your branch and open a pull request on GitHub with a clear summary of changes, motivation, and pre-commit verification outputs.
