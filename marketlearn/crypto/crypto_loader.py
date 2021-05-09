"""Class for Downloading CryptoPrices"""

from __future__ import annotations
from pandas import DataFrame, to_datetime
import ccxt


class CryptoLoader:
    def __init__(self, exchange, symbols, start, end, window):
        self._exchange = exchange
        self._symbols = symbols
        self.start, self.end = start, end
        self.window = window

    def _get_exchange(self):
        try:
            return getattr(ccxt, self._exchange)()
        except AttributeError:
            error_msg = f"Attribution error {self._exchange} not found. \
                \nSupports only binance exchange"
            print(error_msg)

    def load_data(self):
        """loads data from exchange"""
        exchange = self._get_exchange()

        # get ccxt input parameters
        symbols = self._symbols
        start, end, window = self.start, self.end, self.window

        # convert string datetime to milliseconds for ccxt
        start, end = (to_datetime([start, end]).astype(int) / 10 ** 6).astype(int)
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        data = {}
        for s in symbols:
            df = DataFrame(exchange.fetch_ohlcv(s, window, start), columns=cols)
            df["timestamp"] = to_datetime(df["timestamp"], unit="ms")
            data[s] = df
        return data
