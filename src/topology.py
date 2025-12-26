# src/topology.py
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

def corr_distance(rho: float) -> float:
    return float(np.sqrt(2 * (1 - rho)))

def rolling_mst(returns: pd.DataFrame, window: int = 96*14, step: int = 96) -> list:
    """
    returns: columns=assets, index=timestamp, values=logret
    window: ~14 giorni su 15m
    step: aggiornamento giornaliero
    """
    ts = returns.index
    msts = []
    for i in range(window, len(ts), step):
        sub = returns.iloc[i-window:i].dropna(axis=1)
        corr = sub.corr()
        G = nx.Graph()
        for a in corr.columns:
            G.add_node(a)
        for i2, a in enumerate(corr.columns):
            for b in corr.columns[i2+1:]:
                w = corr_distance(corr.loc[a,b])
                G.add_edge(a, b, weight=w, rho=float(corr.loc[a,b]))
        mst = nx.minimum_spanning_tree(G, weight="weight")
        msts.append((ts[i], mst))
    return msts

def mst_node_features(mst: nx.Graph) -> pd.DataFrame:
    deg = dict(mst.degree())
    cen = nx.betweenness_centrality(mst, weight="weight", normalized=True)
    return pd.DataFrame({"degree": deg, "centrality": cen})
