import pandas as pd
import yfinance as yf

def fetch_spy_options_all_expirations():
    """
    Returns a single DataFrame with all SPY calls and puts,
    tagged by type and expiration date.
    """
    ticker = yf.Ticker("SPY")
    expirations = ticker.options
    all_dfs = []

    for exp in expirations:
        chain = ticker.option_chain(exp)
        calls, puts = chain.calls, chain.puts

        calls["Type"]       = "Call"
        calls["Expiration"] = exp

        puts["Type"]        = "Put"
        puts["Expiration"]  = exp

        all_dfs.extend([calls, puts])

    # concatenate and return
    return pd.concat(all_dfs, ignore_index=True)