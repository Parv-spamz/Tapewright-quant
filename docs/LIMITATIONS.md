<!-- GENERATED from ledger head a727d397263cf76aef238a264e59a6a3cc836f2d39a2fb276dc2535dc5d70af5. Do not edit by hand. -->
<!-- GENERATED:BEGIN -->
# Institutional Limitations & Pre-Registered Protocol Biases

## Overview
FinPulse-Apex operates under strict scientific transparency. While mathematical modeling and cryptographic ledgers mitigate data-mining bias, certain structural limitations, sampling constraints, and operational assumptions remain in the research environment.

## 1. Pre-Registered Protocol Known Biases
The active pre-registered protocol explicitly logs the following known biases:
- **Universe Selection Hindsight**: The liquid ETF universe (SPY, QQQ, IWM, EFA, TLT, IEF, GLD, XLE, XLF, XLK, XLV, XLY) was selected using present-day knowledge of longevity and liquidity, excluding ETFs that liquidated or merged during 2005-2024.
- **Total-Return Price Adjustment**: Adjusted close prices use full dividend and split adjustment factors across the historical series. While protecting returns against cash distribution drops, historical price levels reflect future corporate distribution knowledge.
- **Daily Bar Granularity**: Historical equities backtests operate on daily OHLCV bars. Fills occur at next-day open rather than modeling continuous intraday auction queues.
- **Parametric Crypto Friction Assumptions**: Historical crypto trading costs prior to real microstructure recording utilize parameterized exchange tiers and Kyle lambda estimates rather than high-frequency trade-level book fills.

## 2. Market Microstructure & Recording Boundaries
- **1 Hz Polling Snapshot Limit**: The live Level-2 depth recorder captures snapshots and trade streams at 1 Hz intervals via public REST endpoints. Sub-second queue cancellation bursts, cancel-replace sequences, and micro-latency routing advantages cannot be observed at this resolution.
- **ETF-Survivor Universe**: Because the historical dataset comprises liquid ETFs that survived through multiple market cycles, the universe exhibits survivor bias compared to the broad investable equity market.

## 3. Macroeconomic Scenario Modeling
- **Unmeasured Crisis Shock Assumptions**: The macroeconomic stress scenarios (2008 Lehman collapse, 2020 COVID shock, 2022 rate hike cycle) incorporate calibrated historical equity and treasury index drawdowns. However, secondary shocks such as broker margin calls, repo freezes, and sudden liquidity dry-ups remain owner-supplied assumptions rather than directly observed portfolio execution fills.

## 4. Operational & Cryptographic Time Anchors
- **Local-Clock Timestamps**: Local machine clocks on developer workstations cannot serve as an independent cryptographic time authority. Forward testing signals and cryptographic ledger blocks require external independent timestamping (such as GitHub commit timestamps, PR merge logs, or external RFC 3161 time-stamping services) to act as unforgeable chronological proof of forward emission.
<!-- GENERATED:END -->
