from pathlib import Path
import pandas as pd

OUT = Path("data/graphs/causal_lookup.parquet")

def run():
    Path("data/graphs").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(columns=[
        "leader",
        "follower",
        "ete",          # float
        "ste",          # float (opz)
        "best_lag",     # int (barre 15m)
        "ete_hit_rate", # float (0..1)
        "ete_std",      # float
        "asof"          # timestamp di validità (es. snapshot giornaliero)
    ])
    df.to_parquet(OUT, index=False)
    print(f"[OK] creato schema causal -> {OUT}")

if __name__ == "__main__":
    run()
