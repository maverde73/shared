from pathlib import Path
import pandas as pd
import numpy as np

FEAT_DIR = Path("data/features")
OUT_DIR = Path("data/datasets")

def build_tabular(horizon_steps: int = 4) -> pd.DataFrame:
    frames = []
    files = list(FEAT_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Nessuna feature trovata in {FEAT_DIR.resolve()} (esegui features_local.py)")

    for p in files:
        asset = p.stem
        df = pd.read_parquet(p).copy()
        df["asset"] = asset

        # target: ritorno log a t+Δt
        df["target_ret"] = df["logret"].shift(-horizon_steps)

        # target direction (3-class) con soglia semplice basata su ATR
        # soglia = 0.3 * (ATR/close) -> in log approx
        thr = 0.3 * (df["atr14"] / df["close"])
        df["target_dir"] = 0
        df.loc[df["target_ret"] > thr, "target_dir"] = 1
        df.loc[df["target_ret"] < -thr, "target_dir"] = -1

        # feature columns selezionate (minime ma sane)
        keep_cols = [
            "asset",
            "open","high","low","close","volume",
            "logret","vol_1d","rsi14",
            "macd","macd_signal","macd_hist",
            "atr14","vol_z_1d","hl_range",
            "target_ret","target_dir"
        ]
        df = df[keep_cols].dropna()
        frames.append(df)

    all_df = pd.concat(frames).sort_index()

    # aggiungi timestamp come colonna esplicita (utile per salvataggi CSV / debug)
    all_df = all_df.reset_index(names="timestamp")
    return all_df

def run(horizon_steps: int = 4):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = build_tabular(horizon_steps=horizon_steps)
    out_path = OUT_DIR / "tabular.parquet"
    ds.to_parquet(out_path, index=False)
    print(f"[OK] dataset -> {out_path} ({len(ds):,} righe, {ds['asset'].nunique()} asset)")

if __name__ == "__main__":
    run(4)
