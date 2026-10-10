from dataclasses import dataclass, field
import pandas as pd


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)


class ValidationError(Exception):
    """Exception raised when OHLCV or L2 order book data violates schema rules."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__(f"Validation failed with {len(report.errors)} error(s): {report.errors}")


def validate_ohlcv(df: pd.DataFrame, allowed_gap_days: int = 4) -> ValidationReport:
    """Validate that an OHLCV DataFrame conforms to structural and physical invariants.

    Checks:
    - Required columns: open, high, low, close, volume
    - Sorted, unique DatetimeIndex
    - No NaN values
    - No prices <= 0
    - high >= low
    - high >= max(open, close)
    - low <= min(open, close)
    - volume >= 0
    - No date gaps exceeding allowed_gap_days
    """
    errors: list[str] = []

    # 1. Required columns
    required_cols = {"open", "high", "low", "close", "volume"}
    lower_cols = {str(c).lower(): c for c in df.columns}
    missing = required_cols - set(lower_cols.keys())
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
        return ValidationReport(ok=False, errors=errors)

    # 2. DatetimeIndex checks
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append(f"Index must be a pd.DatetimeIndex, got {type(df.index)}")
    else:
        if not df.index.is_monotonic_increasing:
            errors.append("DatetimeIndex is not monotonically increasing.")
        if not df.index.is_unique:
            errors.append("DatetimeIndex contains duplicate timestamps.")

    # 3. Check NaNs
    col_map = {col: lower_cols[col] for col in required_cols}
    for col in required_cols:
        actual_col = col_map[col]
        nan_count = df[actual_col].isna().sum()
        if nan_count > 0:
            errors.append(f"Column '{actual_col}' contains {nan_count} NaN values.")

    # 4. Check prices > 0 and volume >= 0
    for col in ["open", "high", "low", "close"]:
        actual_col = col_map[col]
        non_positive = (df[actual_col] <= 0).sum()
        if non_positive > 0:
            errors.append(f"Column '{actual_col}' contains {non_positive} non-positive values (<= 0).")

    vol_col = col_map["volume"]
    neg_vol = (df[vol_col] < 0).sum()
    if neg_vol > 0:
        errors.append(f"Column '{vol_col}' contains {neg_vol} negative volume values.")

    # 5. OHLC Invariants
    o = df[col_map["open"]]
    h = df[col_map["high"]]
    l = df[col_map["low"]]
    c = df[col_map["close"]]

    if (h < l).any():
        errors.append("Invariant violated: high < low found in bar data.")

    max_oc = pd.concat([o, c], axis=1).max(axis=1)
    if (h < max_oc).any():
        errors.append("Invariant violated: high < max(open, close) found in bar data.")

    min_oc = pd.concat([o, c], axis=1).min(axis=1)
    if (l > min_oc).any():
        errors.append("Invariant violated: low > min(open, close) found in bar data.")

    # 6. Date gaps check
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        diffs = df.index.to_series().diff().dropna()
        max_gap = diffs.max().days
        if max_gap > allowed_gap_days:
            errors.append(
                f"Date gap of {max_gap} days detected, exceeding allowed limit of {allowed_gap_days} days."
            )

    return ValidationReport(ok=(len(errors) == 0), errors=errors)


def assert_valid(df: pd.DataFrame, allowed_gap_days: int = 4) -> None:
    """Validate OHLCV DataFrame and raise ValidationError on failure."""
    report = validate_ohlcv(df, allowed_gap_days=allowed_gap_days)
    if not report.ok:
        raise ValidationError(report)


def validate_book(book_df: pd.DataFrame, levels: int = 5) -> ValidationReport:
    """Validate Level-2 order book structural and monotonic pricing rules.

    Checks:
    - bid_px_k strictly decreasing in k
    - ask_px_k strictly increasing in k
    - best bid < best ask (bid_px_1 < ask_px_1)
    - all level sizes >= 1
    """
    errors: list[str] = []

    if book_df.empty:
        errors.append("Order book DataFrame is empty.")
        return ValidationReport(ok=False, errors=errors)

    # Check best bid < best ask
    if "bid_px_1" in book_df.columns and "ask_px_1" in book_df.columns:
        crossed = (book_df["bid_px_1"] >= book_df["ask_px_1"]).sum()
        if crossed > 0:
            errors.append(f"Found {crossed} instances where best bid >= best ask (crossed/locked book).")
    else:
        errors.append("Missing bid_px_1 or ask_px_1 columns in book.")

    # Check bid price monotonicity (strictly decreasing)
    for k in range(1, levels):
        c_k = f"bid_px_{k}"
        c_next = f"bid_px_{k + 1}"
        if c_k in book_df.columns and c_next in book_df.columns:
            if (book_df[c_k] <= book_df[c_next]).any():
                errors.append(f"Bid prices not strictly decreasing: {c_k} <= {c_next}.")

    # Check ask price monotonicity (strictly increasing)
    for k in range(1, levels):
        c_k = f"ask_px_{k}"
        c_next = f"ask_px_{k + 1}"
        if c_k in book_df.columns and c_next in book_df.columns:
            if (book_df[c_k] >= book_df[c_next]).any():
                errors.append(f"Ask prices not strictly increasing: {c_k} >= {c_next}.")

    # Check sizes >= 1
    for k in range(1, levels + 1):
        for side in ["bid_sz", "ask_sz"]:
            col = f"{side}_{k}"
            if col in book_df.columns:
                if (book_df[col] < 1).any():
                    errors.append(f"Invalid size (< 1) detected in {col}.")

    return ValidationReport(ok=(len(errors) == 0), errors=errors)
