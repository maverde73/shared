from pathlib import Path
import numpy as np
import pandas as pd
import pandas_ta as ta

IN_DIR = Path("data/processed")
OUT_DIR = Path("data/features")

def add_local_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # log returns
    out["logret"] = np.log(out["close"]).diff()

    # rolling vol ~ 1 giorno su 15m (96 barre)
    out["vol_1d"] = out["logret"].rolling(96).std()

    # indicatori
    out["rsi14"] = ta.rsi(out["close"], length=14)
    macd = ta.macd(out["close"])
    out["macd"] = macd["MACD_12_26_9"]
    out["macd_signal"] = macd["MACDs_12_26_9"]
    out["macd_hist"] = macd["MACDh_12_26_9"]

    out["atr14"] = ta.atr(out["high"], out["low"], out["close"], length=14)

    # volume zscore su 1 giorno
    v_mean = out["volume"].rolling(96).mean()
    v_std = out["volume"].rolling(96).std()
    out["vol_z_1d"] = (out["volume"] - v_mean) / v_std

    # spread range normalizzato (micro-volatilità)
    out["hl_range"] = (out["high"] - out["low"]) / out["close"]

    return out.dropna()

def infer_asset_name(filename: str) -> str:
    # BTCUSDT_15min.parquet -> BTCUSDT
    return filename.split("_")[0]

def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = list(IN_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Nessun parquet trovato in {IN_DIR.resolve()} (esegui ingest.py)")

    for p in files:
        asset = infer_asset_name(p.name)
        df = pd.read_parquet(p)
        feat = add_local_features(df)
        out_path = OUT_DIR / f"{asset}.parquet"
        feat.to_parquet(out_path)
        print(f"[OK] features {asset} -> {out_path} ({len(feat):,} righe)")

if __name__ == "__main__":
    run()
