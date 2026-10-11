<div align="center">

<img src="docs/assets/logo.svg" alt="Tapewright-Quant Logo" width="100%" />

# Tapewright-Quant

**FinPulse-Apex · Quantitative Trading Research Engine**

A trading research engine built to simulate markets, test ideas, and challenge its own results.

<p align="center">
  <img src="docs/assets/typing.svg" alt="Simulate markets · Test trading ideas · Challenge the results" width="500" />
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-00E5BF?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+" /></a>
</p>

</div>

## What is Tapewright-Quant?

Tapewright-Quant is an open-source quantitative trading research and educational platform. It brings together market-data processing, trading simulations, portfolio risk tools, and research validation in one Python-based project.

The goal is straightforward: make trading experiments more realistic and make it harder to mistake a promising-looking result for a reliable strategy.

## What does it do?

- **Simulates trading**: Models trading orders, execution costs, and the practical difficulties of getting trades filled.
- **Studies market behaviour**: Provides tools for examining price movements, trading activity, trends, and relationships between assets.
- **Tests trading ideas**: Runs research experiments and evaluates whether strategies hold up beyond their initial results.
- **Examines risk**: Includes tools for portfolio allocation, position sizing, and investigating unfavourable market conditions.
- **Records research evidence**: Uses integrity and verification tools to help trace research experiments and detect changes to recorded artifacts.

## How it works

The research workflow progresses through six stages from raw data ingestion to audited findings:

```mermaid
graph LR
    A["Market Data"] --> B["Data Preparation"]
    B --> C["Trading Ideas"]
    C --> D["Trade Simulation"]
    D --> E["Risk Analysis"]
    E --> F["Research Results"]
```

## Built to question the results

A trading strategy can look successful in a simulation and still fail when its assumptions are tested. Tapewright-Quant includes tools designed to investigate execution costs, unrealistic assumptions, accidental future-data leakage, and overfitting.

Its research process also records experiments and supports reproducibility checks. These controls reduce certain research risks; they do not guarantee that every result is correct or that a strategy will make money.

## Research honesty

The documented research did not produce a strategy that passed every required selection test. The result is reported rather than hidden behind relaxed criteria.

Finding that an idea does not pass is part of research. The next step is stronger evidence, not changing the rules to manufacture a winner.

## Explore the project

- [Project specification](PROJECT_SPEC.md)
- [Research methodology](docs/METHODOLOGY.md)
- [Research findings](docs/FINDINGS.md)
- [Known limitations](docs/LIMITATIONS.md)
- [Forward testing](docs/FORWARD_TEST.md)

## Getting started

### Prerequisites

- Python 3.10 or higher
- Git

### Installation

Clone the repository and install the standard dependencies in a virtual environment:

```bash
git clone https://github.com/Parv-spamz/Tapewright-quant.git
cd Tapewright-quant
python -m venv .venv
```

Activate the virtual environment:

- **Linux / macOS**:
  ```bash
  source .venv/bin/activate
  ```
- **Windows**:
  ```powershell
  .venv\Scripts\activate
  ```

Install requirements:

```bash
pip install -r requirements.txt
```

### Run a verification test

Run a safe configuration test to confirm the environment is set up properly:

```bash
pytest tests/unit/test_config.py
```

Explore the code, read the methodology, and examine the evidence. Contributions and thoughtful questions are welcome.
