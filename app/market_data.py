import yfinance as yf

def get_snapshot(ticker: str):
    t = yf.Ticker(ticker)
    info = t.fast_info
    return {"last": info.get("last_price"), "currency": info.get("currency")}
