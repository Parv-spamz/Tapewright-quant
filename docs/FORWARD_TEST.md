<!-- GENERATED from ledger head a727d397263cf76aef238a264e59a6a3cc836f2d39a2fb276dc2535dc5d70af5. Do not edit by hand. -->
<!-- GENERATED:BEGIN -->
# Forward Testing Operational Protocol (Windows Task Scheduler)

## Overview
This document specifies the operational procedure for executing automated forward testing signals and evaluation routines via Windows Task Scheduler.
All forward test records are cryptographically pre-registered and anchored to the project ledger.

## Prerequisites
- Python virtual environment activated with project dependencies installed.
- Working directory set to the project root containing `research/` and `integrity/`.
- Valid candidate strategy pre-registered with `role='selected'` in the ledger.

## Task 1: Daily Forward Signal Emission
**Objective**: Emits candidate strategy trading signals for the next trading bar. Must execute strictly BEFORE the target bar opens.

1. Open **Windows Task Scheduler** (`taskschd.msc`).
2. Click **Create Task** in the Actions pane.
3. On the **General** tab:
   - Name: `Tapewright_Forward_Emit_Signals`
   - Security options: Run whether user is logged on or not / Run with highest privileges.
4. On the **Triggers** tab:
   - Click **New...**
   - Begin the task: **On a schedule**
   - Settings: **Daily**, recurring every day.
   - Time: After US market close (e.g. 17:00 EST / 21:00 UTC).
5. On the **Actions** tab:
   - Action: **Start a program**
   - Program/script: `python` (or full path to virtual environment python executable)
   - Add arguments: `-m research.forward emit --spec-hash <SELECTED_SPEC_HASH>`
   - Start in: `<PROJECT_ROOT_DIRECTORY>`
6. On the **Settings** tab:
   - Check: **Do not start a new instance** if the task is already running.

## Task 2: Daily Forward Outcome Evaluation
**Objective**: Evaluates realized execution returns for completed forward signals after the target bar has fully closed.

1. In **Windows Task Scheduler**, click **Create Task**.
2. On the **General** tab:
   - Name: `Tapewright_Forward_Evaluate_Outcomes`
3. On the **Triggers** tab:
   - Click **New...**
   - Settings: **Daily**, recurring every day.
   - Time: Subsequent day after bar completion (e.g. 17:15 EST / 21:15 UTC).
4. On the **Actions** tab:
   - Program/script: `python`
   - Add arguments: `-m research.forward evaluate`
   - Start in: `<PROJECT_ROOT_DIRECTORY>`

## Task 3: Cryptographic Ledger Anchoring
**Objective**: Verifies the ledger hash chain, logs chronological anchor points, and outputs outside timestamping instructions.

1. In **Windows Task Scheduler**, click **Create Task**.
2. On the **General** tab:
   - Name: `Tapewright_Forward_Anchor_Ledger`
3. On the **Triggers** tab:
   - Settings: **Daily**, following signal emission.
4. On the **Actions** tab:
   - Program/script: `python`
   - Add arguments: `-m integrity.anchor`
   - Start in: `<PROJECT_ROOT_DIRECTORY>`

## Manual Verification
To review the cumulative forward equity and report without scheduling:
```powershell
python -m research.forward report
```
To run integrity stage verification across all forward records:
```powershell
python -m integrity.check_ledger
```
<!-- GENERATED:END -->
