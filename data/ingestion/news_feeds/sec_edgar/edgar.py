import json
import os
import urllib.error
import urllib.request
import pandas as pd

from data.ingestion.market_feeds.equities.yahoo import FeedError


def parse_edgar_submissions(raw_text: str) -> pd.DataFrame:
    """Parse SEC EDGAR company submissions JSON text into a DataFrame of filings metadata."""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise FeedError(f"Failed to decode EDGAR submissions JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise FeedError("Invalid EDGAR response: expected top-level JSON object.")

    filings = data.get("filings", {})
    recent = filings.get("recent", {})

    if not recent or not isinstance(recent, dict):
        return pd.DataFrame()

    df = pd.DataFrame(recent)
    if "filingDate" in df.columns:
        df["filing_date"] = pd.to_datetime(df["filingDate"])
        df = df.set_index("filing_date")
        df = df.sort_index(ascending=False)

    return df


def fetch_edgar_submissions(
    cik: str,
    timeout: float = 10.0,
) -> pd.DataFrame:
    """Fetch company filings submissions from SEC EDGAR API.

    Requires an institutional User-Agent specified in TAPEWRIGHT_USER_AGENT.
    """
    clean_cik = str(cik).strip().zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{clean_cik}.json"

    user_agent = os.environ.get(
        "TAPEWRIGHT_USER_AGENT",
        "TapewrightQuant/1.0 (research@tapewright.org)",
    )

    req = urllib.request.Request(
        url,
        headers={"User-Agent": user_agent, "Accept-Encoding": "identity"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw_text = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FeedError(f"Failed to fetch EDGAR submissions for CIK {cik}: {exc}") from exc

    return parse_edgar_submissions(raw_text)
