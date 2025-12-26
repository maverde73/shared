import numpy as np
import pandas as pd
from pathlib import Path

FEE_BPS = 5      # 0.05%
SLIP_BPS = 5     # 0.05%
THR = 0.0008     # soglia sul log-return previsto (tarare su validation)

def cost_per_trade():
    return (FEE_BPS + SLIP_BPS) / 10000.0

def run(pred_df: pd.DataFrame):
    """
    pred_df deve avere:
      timestamp, asset, close, target_ret, pred
    """
    df = pred_df.sort_values(["asset","timestamp"]).copy()
    c = cost_per_trade()

    # PnL per trade: sign(pred)*real_ret - cost
    df["pos"] = 0
    df.loc[df["pred"] > THR, "pos"] = 1
    df.loc[df["pred"] < -THR, "pos"] = -1

    df["trade"] = (df["pos"] != 0).astype(int)
    df["pnl"] = df["pos"] * df["target_ret"] - df["trade"] * c

    # metriche base
    pnl = df["pnl"].dropna()
    cum = pnl.cumsum()
    dd = cum - cum.cummax()

    out = {
        "trades": int(df["trade"].sum()),
        "avg_pnl": float(pnl.mean()),
        "std_pnl": float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
        "sharpe_like": float(pnl.mean() / (pnl.std(ddof=1) + 1e-12)),
        "max_drawdown": float(dd.min()),
    }
    return out

if __name__ == "__main__":
    print("Questo modulo si usa dopo aver generato predizioni nel test fold.")
