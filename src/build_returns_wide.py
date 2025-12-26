from pathlib import Path
import pandas as pd

FEAT_DIR = Path("data/features")
OUT_PATH = Path("data/graphs/returns_wide.parquet")

def run():
    Path("data/graphs").mkdir(parents=True, exist_ok=True)

    frames = []
    for p in FEAT_DIR.glob("*.parquet"):
        asset = p.stem
        df = pd.read_parquet(p)[["logret"]].copy()
        df = df.rename(columns={"logret": asset})
        frames.append(df)

    if not frames:
        raise FileNotFoundError("Nessun file in data/features/*.parquet")

    wide = pd.concat(frames, axis=1).sort_index()
    # mantieni solo timestamp dove hai un minimo di copertura (es. almeno 80% asset)
    min_cols = int(0.8 * wide.shape[1])
    wide = wide.dropna(thresh=min_cols)

    wide.to_parquet(OUT_PATH)
    print(f"[OK] returns_wide -> {OUT_PATH}  shape={wide.shape}")

if __name__ == "__main__":
    run()
