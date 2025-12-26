from pathlib import Path
import pandas as pd
import numpy as np

FEAT_DIR = Path("data/features")
NODE_PATH = Path("data/graphs/mst_node_features.parquet")
GLOB_PATH = Path("data/graphs/mst_global_features.parquet")
CAUSAL_PATH = Path("data/graphs/causal_lookup.parquet")

OUT = Path("data/datasets/tabular_full.parquet")

H_30M = 2  # 2*15m
H_1H  = 4  # 4*15m

LEADERS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]

def run():
    Path("data/datasets").mkdir(parents=True, exist_ok=True)

    node = pd.read_parquet(NODE_PATH)
    glob = pd.read_parquet(GLOB_PATH)

    node["timestamp"] = pd.to_datetime(node["timestamp"], utc=True)
    glob["timestamp"] = pd.to_datetime(glob["timestamp"], utc=True)

    # global features indicizzate per snapshot
    glob = glob.sort_values("timestamp").set_index("timestamp")

    # node features per snapshot+asset
    node = node.sort_values("timestamp")

    # causal (opzionale)
    causal = None
    if CAUSAL_PATH.exists():
        causal = pd.read_parquet(CAUSAL_PATH)
        if len(causal) == 0:
            causal = None
        else:
            causal["asof"] = pd.to_datetime(causal["asof"], utc=True)

    frames = []
    for p in FEAT_DIR.glob("*.parquet"):
        asset = p.stem
        df = pd.read_parquet(p).copy()
        df = df.sort_index()
        df["timestamp"] = df.index
        df["asset"] = asset

        # targets
        df["target_ret_30m"] = df["logret"].shift(-H_30M)
        df["target_ret_1h"] = df["logret"].shift(-H_1H)

        # direction labels con soglia ATR/close
        thr = 0.3 * (df["atr14"] / df["close"])
        for hname in ["30m", "1h"]:
            tcol = f"target_ret_{hname}"
            dcol = f"target_dir_{hname}"
            df[dcol] = 0
            df.loc[df[tcol] > thr, dcol] = 1
            df.loc[df[tcol] < -thr, dcol] = -1

        # join MST node features: merge_asof su timestamp per snapshot giornaliero
        node_a = node[node["asset"] == asset][["timestamp","degree","centrality_btw","dist_from_btc"]].copy()
        node_a = node_a.sort_values("timestamp")

        df = pd.merge_asof(
            df.sort_values("timestamp"),
            node_a,
            on="timestamp",
            direction="backward"
        )

        # join MST global features: merge_asof
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            glob.reset_index().rename(columns={"index":"timestamp"}).sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )

        # causal features per leader (se disponibili): merge_asof su asof
        if causal is not None:
            for L in LEADERS:
                sub = causal[(causal["leader"] == L) & (causal["follower"] == asset)][
                    ["asof","ete","best_lag","ete_hit_rate","ete_std"]
                ].sort_values("asof")
                if len(sub) == 0:
                    df[f"ete_from_{L}"] = np.nan
                    df[f"lag_from_{L}"] = np.nan
                    df[f"ete_hr_from_{L}"] = np.nan
                    continue
                sub = sub.rename(columns={
                    "asof": "timestamp",
                    "ete": f"ete_from_{L}",
                    "best_lag": f"lag_from_{L}",
                    "ete_hit_rate": f"ete_hr_from_{L}",
                    "ete_std": f"ete_std_from_{L}",
                })
                df = pd.merge_asof(
                    df.sort_values("timestamp"),
                    sub.sort_values("timestamp"),
                    on="timestamp",
                    direction="backward"
                )

        # selezione colonne finali
        keep = [
            "timestamp","asset",
            "open","high","low","close","volume",
            "logret","vol_1d","rsi14",
            "macd","macd_signal","macd_hist",
            "atr14","vol_z_1d","hl_range",
            "degree","centrality_btw","dist_from_btc",
            "centralization_deg","avg_edge_rho",
            "target_ret_30m","target_dir_30m",
            "target_ret_1h","target_dir_1h",
        ]
        # aggiungi eventuali causal cols se presenti
        if causal is not None:
            for L in LEADERS:
                keep += [f"ete_from_{L}", f"lag_from_{L}", f"ete_hr_from_{L}", f"ete_std_from_{L}"]

        df = df[keep].dropna(subset=["target_ret_30m","target_ret_1h"])
        frames.append(df)

    out = pd.concat(frames).sort_values(["timestamp","asset"])
    out.to_parquet(OUT, index=False)
    print(f"[OK] tabular_full -> {OUT} rows={len(out):,} assets={out['asset'].nunique()}")

if __name__ == "__main__":
    run()
