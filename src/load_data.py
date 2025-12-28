import pandas as pd
import os

def load_file(path):
    if path.endswith(".csv"):
        return pd.read_csv(path, parse_dates=["Date"])
    elif path.endswith(".xlsx"):
        return pd.read_excel(path, parse_dates=["Date"])
    else:
        raise ValueError("Unsupported file format")

def load_stock(symbol, base_path="data/raw/stocks"):
    for ext in ["csv", "xlsx"]:
        path = f"{base_path}/{symbol}.{ext}"
        if os.path.exists(path):
            df = load_file(path)
            break
    else:
        raise FileNotFoundError(f"No file found for {symbol}")

    df = df.sort_values("Date").reset_index(drop=True)
    df["ret"] = df["Adj Close"].pct_change()
    return df.dropna().reset_index(drop=True)

def load_etf(symbol, base_path="data/raw/etfs"):
    for ext in ["csv", "xlsx"]:
        path = f"{base_path}/{symbol}.{ext}"
        if os.path.exists(path):
            df = load_file(path)
            break
    else:
        raise FileNotFoundError(f"No ETF file found for {symbol}")

    df = df.sort_values("Date").reset_index(drop=True)
    df["ret"] = df["Adj Close"].pct_change()
    return df.dropna().reset_index(drop=True)
