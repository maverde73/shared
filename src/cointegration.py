# src/cointegration.py (minimo utile)
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_spread(prices: pd.DataFrame, det_order=0, k_ar_diff=1):
    """
    prices: columns=[asset1, asset2, ...], index=timestamp
    """
    logp = np.log(prices).dropna()
    res = coint_johansen(logp.values, det_order, k_ar_diff)
    vec = res.evec[:, 0]  # primo vettore
    spread = logp.values @ vec
    return pd.Series(spread, index=logp.index), vec

def zscore(s: pd.Series, window: int = 96*14) -> pd.Series:
    m = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - m) / sd
