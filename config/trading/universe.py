from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    symbol: str
    asset_class: str
    currency: str
    exchange: str
    calendar: str


DEFAULT_UNIVERSE: tuple[Asset, ...] = (
    # 6 Equities
    Asset(symbol="AAPL", asset_class="EQUITY", currency="USD", exchange="NASDAQ", calendar="NYSE"),
    Asset(symbol="MSFT", asset_class="EQUITY", currency="USD", exchange="NASDAQ", calendar="NYSE"),
    Asset(symbol="JPM", asset_class="EQUITY", currency="USD", exchange="NYSE", calendar="NYSE"),
    Asset(symbol="XOM", asset_class="EQUITY", currency="USD", exchange="NYSE", calendar="NYSE"),
    Asset(symbol="JNJ", asset_class="EQUITY", currency="USD", exchange="NYSE", calendar="NYSE"),
    Asset(symbol="SPY", asset_class="EQUITY", currency="USD", exchange="ARCA", calendar="NYSE"),
    # 2 Crypto
    Asset(symbol="BTC-USD", asset_class="CRYPTO", currency="USD", exchange="BINANCE", calendar="CRYPTO_24_7"),
    Asset(symbol="ETH-USD", asset_class="CRYPTO", currency="USD", exchange="BINANCE", calendar="CRYPTO_24_7"),
    # 2 FX
    Asset(symbol="EURUSD", asset_class="FX", currency="USD", exchange="FX_SPOT", calendar="NYSE"),
    Asset(symbol="USDJPY", asset_class="FX", currency="JPY", exchange="FX_SPOT", calendar="NYSE"),
)


def get_universe(asset_class: str | None = None) -> list[Asset]:
    """Return assets in DEFAULT_UNIVERSE, optionally filtered by asset_class.

    If asset_class is None, returns all assets.
    The asset_class match is case-insensitive.
    """
    if asset_class is None:
        return list(DEFAULT_UNIVERSE)

    target_class = asset_class.strip().upper()
    return [asset for asset in DEFAULT_UNIVERSE if asset.asset_class.upper() == target_class]
