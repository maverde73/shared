from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

def load_ohlcv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    return df

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    return out.dropna()

def infer_asset_name(filename: str) -> str:
    # ADAUSDT_5m.csv -> ADAUSDT
    return filename.split("_")[0]

def run(rule: str = "15min"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = list(RAW_DIR.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"Nessun CSV trovato in {RAW_DIR.resolve()}")

    for p in paths:
        asset = infer_asset_name(p.name)
        df = load_ohlcv(p)
        df = resample_ohlcv(df, rule)
        out_path = OUT_DIR / f"{asset}_{rule}.parquet"
        df.to_parquet(out_path)
        print(f"[OK] {asset} -> {out_path} ({len(df):,} righe)")

if __name__ == "__main__":
    run("15min")
