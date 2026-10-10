import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd

from data.ingestion.market_feeds.equities.yahoo import FeedError


def parse_rss_feed(raw_text: str) -> pd.DataFrame:
    """Parse RSS 2.0 XML text into a standardized news headlines DataFrame.

    Normalizes RFC822 and ISO-8601 published timestamps to UTC.
    """
    try:
        root = ET.fromstring(raw_text)
    except ET.ParseError as exc:
        raise FeedError(f"Failed to parse RSS XML: {exc}") from exc

    # Find channel
    channel = root.find("channel")
    if channel is None:
        channel = root

    channel_title = channel.findtext("title", default="Unknown RSS")

    items = channel.findall("item")
    if not items:
        # Check Atom feed fallback: entry tags
        items = root.findall("{http://www.w3.org/2005/Atom}entry")

    records: list[dict[str, Any]] = []

    for item in items:
        title = item.findtext("title", default="").strip()
        link = item.findtext("link", default="").strip()
        summary = item.findtext("description", default="") or item.findtext("summary", default="")
        pub_date_str = item.findtext("pubDate", default="") or item.findtext("published", default="") or item.findtext("updated", default="")

        if pub_date_str:
            try:
                published_at = pd.to_datetime(pub_date_str.strip(), utc=True)
            except Exception:
                published_at = pd.NaT
        else:
            published_at = pd.NaT

        records.append(
            {
                "title": title,
                "link": link,
                "summary": summary.strip(),
                "published_at": published_at,
                "source": channel_title,
            }
        )

    if not records:
        return pd.DataFrame(columns=["title", "link", "summary", "published_at", "source"])

    df = pd.DataFrame(records)
    return df


def fetch_rss_feed(url: str, timeout: float = 10.0) -> pd.DataFrame:
    """Fetch and parse financial RSS feed via HTTP GET."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw_text = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FeedError(f"Failed to fetch RSS feed from {url}: {exc}") from exc

    return parse_rss_feed(raw_text)
