import pandas as pd
from dataclasses import dataclass

@dataclass
class WFConfig:
    train_days: int = 365 * 2      # 2 anni
    val_days: int = 90             # 3 mesi
    test_days: int = 90            # 3 mesi
    step_days: int = 90            # avanzamento

def walk_forward_splits(df: pd.DataFrame, cfg: WFConfig):
    # df deve avere colonna timestamp (UTC)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    start = df["timestamp"].min()
    end = df["timestamp"].max()

    cur = start
    while True:
        train_end = cur + pd.Timedelta(days=cfg.train_days)
        val_end = train_end + pd.Timedelta(days=cfg.val_days)
        test_end = val_end + pd.Timedelta(days=cfg.test_days)

        if test_end > end:
            break

        train_idx = (df["timestamp"] >= cur) & (df["timestamp"] < train_end)
        val_idx   = (df["timestamp"] >= train_end) & (df["timestamp"] < val_end)
        test_idx  = (df["timestamp"] >= val_end) & (df["timestamp"] < test_end)

        yield df[train_idx], df[val_idx], df[test_idx]

        cur = cur + pd.Timedelta(days=cfg.step_days)
