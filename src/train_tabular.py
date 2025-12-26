import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

from split import WFConfig, walk_forward_splits

DATASET_PATH = Path("data/datasets/tabular.parquet")

FEATURES = [
    "logret","vol_1d","rsi14",
    "macd","macd_signal","macd_hist",
    "atr14","vol_z_1d","hl_range",
]
TARGET = "target_ret"

def prep_xy(df: pd.DataFrame):
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(float)
    return X, y

def run():
    ds = pd.read_parquet(DATASET_PATH)
    cfg = WFConfig()

    fold = 0
    reports = []

    for train_df, val_df, test_df in walk_forward_splits(ds, cfg):
        fold += 1

        Xtr, ytr = prep_xy(train_df)
        Xva, yva = prep_xy(val_df)
        Xte, yte = prep_xy(test_df)

        model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=400,
            l2_regularization=1e-3,
            random_state=42
        )
        model.fit(Xtr, ytr)

        pred_va = model.predict(Xva)
        pred_te = model.predict(Xte)

        rmse_va = float(np.sqrt(mean_squared_error(yva, pred_va)))
        rmse_te = float(np.sqrt(mean_squared_error(yte, pred_te)))

        # “hit-rate” direzionale grezzo (segno corretto)
        hit_te = float((np.sign(pred_te) == np.sign(yte.values)).mean())

        # importance su validation (più corretto del test)
        pi = permutation_importance(model, Xva, yva, n_repeats=3, random_state=42)
        imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)

        reports.append({
            "fold": fold,
            "train_start": train_df["timestamp"].min(),
            "train_end": train_df["timestamp"].max(),
            "rmse_val": rmse_va,
            "rmse_test": rmse_te,
            "hit_test": hit_te,
            "top_features": imp.head(5).to_dict()
        })

        print(f"\n=== FOLD {fold} ===")
        print(f"RMSE val : {rmse_va:.6f}")
        print(f"RMSE test: {rmse_te:.6f}")
        print(f"Hit test : {hit_te:.3f}")
        print("Top5 features:", imp.head(5).to_dict())

    out = Path("data/datasets/reports.json")
    pd.DataFrame(reports).to_json(out, orient="records", date_format="iso")
    print(f"\n[OK] Report salvato in {out}")

if __name__ == "__main__":
    run()
