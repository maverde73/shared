from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

RET_PATH = Path("data/graphs/returns_wide.parquet")
OUT_DIR = Path("data/graphs")

BARS_PER_DAY = 96  # 15m
WINDOW = BARS_PER_DAY * 14
STEP = BARS_PER_DAY

def corr_distance(rho: float) -> float:
    return float(np.sqrt(2 * (1 - rho)))

def mst_from_corr(corr: pd.DataFrame) -> nx.Graph:
    cols = list(corr.columns)
    G = nx.Graph()
    G.add_nodes_from(cols)
    for i, a in enumerate(cols):
        for b in cols[i+1:]:
            rho = float(corr.loc[a, b])
            if np.isnan(rho):
                continue
            G.add_edge(a, b, weight=corr_distance(rho), rho=rho)
    return nx.minimum_spanning_tree(G, weight="weight")

def centralization_degree(mst: nx.Graph) -> float:
    # centralizzazione (0..1) stile Freeman su degree
    degs = np.array([d for _, d in mst.degree()], dtype=float)
    n = len(degs)
    if n <= 2:
        return 0.0
    dmax = degs.max()
    num = np.sum(dmax - degs)
    den = (n - 1) * (n - 2)
    return float(num / den) if den > 0 else 0.0

def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ret = pd.read_parquet(RET_PATH).sort_index()
    ts = ret.index

    node_rows = []
    glob_rows = []
    edge_counts = {}  # (a,b) -> count

    for i in range(WINDOW, len(ts), STEP):
        t_snap = ts[i]
        sub = ret.iloc[i-WINDOW:i]
        sub = sub.dropna(axis=1)  # asset presenti nella finestra
        if sub.shape[1] < 8:
            continue

        corr = sub.corr()
        mst = mst_from_corr(corr)

        # node features
        deg = dict(mst.degree())
        btw = nx.betweenness_centrality(mst, weight="weight", normalized=True)
        # distanza da BTC se esiste nel grafo
        dist_btc = {}
        if "BTCUSDT" in mst.nodes:
            lengths = nx.single_source_dijkstra_path_length(mst, "BTCUSDT", weight="weight")
            dist_btc = {k: float(v) for k, v in lengths.items()}
        else:
            dist_btc = {k: np.nan for k in mst.nodes}

        for a in mst.nodes:
            node_rows.append({
                "timestamp": t_snap,
                "asset": a,
                "degree": float(deg.get(a, 0)),
                "centrality_btw": float(btw.get(a, 0.0)),
                "dist_from_btc": float(dist_btc.get(a, np.nan)),
            })

        # global features
        glob_rows.append({
            "timestamp": t_snap,
            "n_assets": int(mst.number_of_nodes()),
            "centralization_deg": centralization_degree(mst),
            "avg_edge_rho": float(np.mean([mst.edges[e]["rho"] for e in mst.edges])),
        })

        # edge persistence counts
        for a, b in mst.edges:
            key = tuple(sorted((a, b)))
            edge_counts[key] = edge_counts.get(key, 0) + 1

        print(f"[MST] {t_snap}  nodes={mst.number_of_nodes()}")

    node_df = pd.DataFrame(node_rows)
    glob_df = pd.DataFrame(glob_rows)

    # persistenza archi normalizzata
    n_snaps = glob_df.shape[0]
    edge_rows = []
    for (a, b), c in edge_counts.items():
        edge_rows.append({
            "asset_a": a,
            "asset_b": b,
            "count": int(c),
            "persistence": float(c / n_snaps) if n_snaps else 0.0
        })
    edge_df = pd.DataFrame(edge_rows).sort_values("persistence", ascending=False)

    node_out = OUT_DIR / "mst_node_features.parquet"
    glob_out = OUT_DIR / "mst_global_features.parquet"
    edge_out = OUT_DIR / "edge_persistence.parquet"

    node_df.to_parquet(node_out, index=False)
    glob_df.to_parquet(glob_out, index=False)
    edge_df.to_parquet(edge_out, index=False)

    print(f"[OK] node_features -> {node_out} rows={len(node_df):,}")
    print(f"[OK] global_features -> {glob_out} rows={len(glob_df):,}")
    print(f"[OK] edge_persistence -> {edge_out} rows={len(edge_df):,}")

if __name__ == "__main__":
    run()
