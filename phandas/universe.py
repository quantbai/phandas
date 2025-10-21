import requests
import pandas as pd

""" Universe Class to Get Top Cryptocurrencies with filter (optional) by Market Cap & Volume """
class Universe:
    def __init__(self, 
                 top: int = 100, 
                 market_cap_threshold: int | None = None, 
                 volume_threshold: int | None = None, 
                 # Columns to keep in the final DataFrame, if it is None keep all the default columns
                 keep_columns: list[str] | None = None,
                 vs_currency: str = "usd"):
        
        self.top = top
        self.market_cap_threshold = market_cap_threshold
        self.volume_threshold = volume_threshold
        self.keep_columns = keep_columns
        self.vs_currency = vs_currency
        self.coins = self.top_coins(
            top=top, market_cap_threshold=market_cap_threshold, volume_threshold=volume_threshold,
            keep_columns=keep_columns
        )

    """ Filter Functions """
    @staticmethod
    def is_stable(name: str, symbol: str) -> bool:
        stable_keywords = ["usd", "usdt", "usdc", "busd", "dai", "ust", "tusd"]
        name = name.lower()
        symbol = symbol.lower()
        return any(k in name for k in stable_keywords) or any(k in symbol for k in stable_keywords)

    @staticmethod
    def is_wrapped(name: str, symbol: str) -> bool:
        wrapped_keywords = ["wrap", "wrapped", "staked", "bridged", "pool", "pegged", "synthetic"]
        name = name.lower()
        symbol = symbol.lower()
        return any(k in name for k in wrapped_keywords) or any(k in symbol for k in wrapped_keywords)

    @staticmethod
    def is_fake_or_clone(name: str, symbol: str) -> bool:
        name = name.lower()
        symbol = symbol.lower()
        legit = {
            "bitcoin": ["btc"],
            "ethereum": ["eth"],
        }
        banned_keywords = ["bridge", "bridged", "pegged", "synthetic", "protocol", "solv", "rebased", "v2", "2.0"]
        for legit_name, legit_symbols in legit.items():
            if any(base in name for base in legit_symbols) and legit_name not in name:
                return True
            if any(base in symbol for base in legit_symbols) and legit_name not in name:
                return True
        if any(k in name for k in banned_keywords):
            return True
        return False

    """ Main Function to Get Top Coins """
    def top_coins(self, 
                top: int = 100, 
                market_cap_threshold: int | None = None, 
                volume_threshold: int | None = None,
                keep_columns: list[str] | None = None) -> pd.DataFrame:

        coins = []
        # CoinGecko API per_page max is 250
        per_page = top if top < 250 else 250
        total_pages = (top // per_page) + 1

        for page in range(1, total_pages + 1):
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": self.vs_currency,
                "order": "market_cap_desc",
                "per_page": per_page,
                "page": page,
                "sparkline": False
            }
            r = requests.get(url, params=params)
            if r.status_code != 200:
                print(f"⚠️ CoinGecko API '{page}' page error, status code: {r.status_code}")
                continue
            coins.extend(r.json())

        """ Filter Stable & Wrapped Coin """
        df = pd.DataFrame(coins)
        df["is_stable"] = df.apply(lambda x: self.is_stable(x["name"], x["symbol"]), axis=1)
        df["is_wrapped"] = df.apply(lambda x: self.is_wrapped(x["name"], x["symbol"]), axis=1)
        df["is_fake"] = df.apply(lambda x: self.is_fake_or_clone(x["name"], x["symbol"]), axis=1)
        df = df[(df["is_stable"] == False) & (df["is_wrapped"] == False) & (df["is_fake"] == False)]

        """ Filter by Market Cap & Volume """
        if market_cap_threshold is not None:
            df = df[df["market_cap"] >= market_cap_threshold]
        if volume_threshold is not None:
            df = df[df["total_volume"] >= volume_threshold]

        df = df.sort_values("market_cap", ascending=False).head(top).reset_index(drop=True)
        if keep_columns is not None:
            df = df[keep_columns]

        return df
    
    """ Convert self.coins to symbols list with upper case """
    def to_symbols(self) -> list[str]:
        return [symbol.upper() for symbol in self.coins["symbol"].tolist()]