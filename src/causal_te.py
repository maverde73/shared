# src/causal_te.py (scheletro)
import pandas as pd
from pathlib import Path

def compute_ste_ete(leader_series: pd.Series, follower_series: pd.Series,
                    max_lag: int = 12, n_surrogates: int = 50) -> dict:
    """
    Ritorna:
      best_lag, ste, ete
    Implementazione: STE (ordinal patterns) + ETE (surrogati shuffled)
    """
    # TODO: implementazione concreta
    raise NotImplementedError

def build_causal_lookup(features_dir="data/features", out_path="data/graphs/causal_lookup.parquet"):
    Path("data/graphs").mkdir(parents=True, exist_ok=True)
    assets = [p.stem for p in Path(features_dir).glob("*.parquet")]
    # output: leader, follower, ste, ete, best_lag, stability_score
    rows = []
    # TODO: loop su coppie + rolling windows per stability
    pd.DataFrame(rows).to_parquet(out_path)
