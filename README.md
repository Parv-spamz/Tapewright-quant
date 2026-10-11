<!-- GENERATED from ledger head a727d397263cf76aef238a264e59a6a3cc836f2d39a2fb276dc2535dc5d70af5. Do not edit by hand. -->
<!-- GENERATED:BEGIN -->
# FinPulse-Apex: Institutional Quantitative Research & Evidence Engine

## Overview
FinPulse-Apex (Tapewright) is an institutional-grade, zero-lookahead, causal quantitative trading and research engine. The architecture adheres to rigorous anti-overfitting discipline, cryptographic ledger provenance, mutation-tested unit suites, metamorphic testing, and empirical microstructure calibration.

## Architecture Highlights
- **Strictly Causal Discrete-Event Engine**: Discrete event loop eliminating future-data peeking. All events process chronological bars with point-in-time state updates.
- **Microstructure-Realistic Execution**: Multi-tier order books, Kyle's lambda market impact, square-root slippage, and queue position decay modeling.
- **Lopez de Prado Machine Learning Discipline**: Purged K-Fold and Walk-Forward cross-validation, combinatorial embargoing, deflated Sharpe ratio (DSR), and CSCV probability of backtest overfitting (PBO).
- **Cryptographic Provenance Ledger**: Every dataset, protocol registration, experiment spec, and test execution is hashed (SHA-256) and appended to a tamper-evident audit ledger.
- **Offline Reproducibility**: 100% deterministic offline reproduction of data stores from content-addressed raw archives and recorded backtest specs.

## Verification & Hardening Metrics

### Test Suite Mutation Hardening
- Event Loop Engine: 84.0% kill rate
- Execution Models: 100.0% kill rate
- Risk Measures & Anti-Overfit: 80.0% kill rate
- Fractional Differentiation: 100.0% kill rate
- Feature Registry: 100.0% kill rate
- Training Pipelines: 100.0% kill rate
- Signal Generation: 80.0% kill rate

### Research Discipline & Overfitting Statistics
- Total Parameter Trials Recorded on Dev: 55
- Probability of Backtest Overfitting (PBO): 0.1580
- Queue Simulator Agreement vs Real Binance L2: 100.0%

## Quickstart & Verification Commands

### Environment Setup
Activate your Python environment and verify constraints:
```powershell
python scripts/check_syntax.py
python scripts/check_constraints.py
python scripts/check_test_integrity.py
```

### Full Integrity Verification Gate
Run the cryptographic integrity verification gate across all stages:
```powershell
python -m integrity.verify_all --lock-sha <LOCK_FILE_SHA256>
```

### 100% Deterministic Offline Reproduction
Rebuild all datasets from raw content-addressed archive and reproduce baseline, trial, and selected validation experiments:
```powershell
python -m research.reproduce
```

### Interactive Evidence Dashboard
Boot the headless interactive Streamlit dashboard:
```powershell
streamlit run backtesting/reports/tearsheet/app.py
```

## Documentation Sitemap
- [FINDINGS.md](docs/FINDINGS.md): Comprehensive empirical findings, validation survival, ablations, stress tests, and calibration metrics.
- [METHODOLOGY.md](docs/METHODOLOGY.md): Mathematical derivations, causal gating, purged cross-validation, and execution mechanics.
- [LIMITATIONS.md](docs/LIMITATIONS.md): Transparent record of known protocol biases, dataset limits, and unmeasured assumptions.
- [FORWARD_TEST.md](docs/FORWARD_TEST.md): Operational instructions for Windows Task Scheduler paper forward testing.
<!-- GENERATED:END -->
