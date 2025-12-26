from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
from joblib import Parallel, delayed
import os

RET_PATH = Path("data/graphs/returns_wide.parquet")
OUT_PATH = Path("data/graphs/causal_lookup.parquet")

LEADERS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]

BAR_MINUTES = 15
LAGS = list(range(1, 13))          # 15m..3h
SHORT_WINDOW_HOURS = 6
SHORT_BARS = int((SHORT_WINDOW_HOURS * 60) / BAR_MINUTES)  # 24
LONG_WINDOW_DAYS = 14

SURROGATES = 20    # <<< abbassa a 20: enorme speedup, buona stabilità
Z_TH = 2.0

M = 3
TAU = 1
ASOF_TIME = "23:45"

RNG_SEED = 42


def ordinal_symbols(x: np.ndarray, m: int = M, tau: int = TAU) -> np.ndarray:
    n = len(x)
    span = (m - 1) * tau
    if n <= span:
        return np.array([], dtype=np.int16)

    syms = []
    for t in range(span, n):
        w = x[t - span : t + 1 : tau]
        jitter = (np.arange(m) * 1e-12)
        order = tuple(np.argsort(w + jitter))
        syms.append(order)

    uniq = {p: i for i, p in enumerate(sorted(set(syms)))}
    return np.array([uniq[p] for p in syms], dtype=np.int16)


def symbolic_te(x_sym: np.ndarray, y_sym: np.ndarray, lag: int) -> float:
    n = min(len(x_sym), len(y_sym))
    start = max(1, lag)
    if n <= start + 2:
        return 0.0

    c_y_y1_x = Counter()
    c_y1_x = Counter()
    c_y_y1 = Counter()
    c_y1 = Counter()

    for i in range(start, n):
        yt = int(y_sym[i])
        y1 = int(y_sym[i - 1])
        xl = int(x_sym[i - lag])

        c_y_y1_x[(yt, y1, xl)] += 1
        c_y1_x[(y1, xl)] += 1
        c_y_y1[(yt, y1)] += 1
        c_y1[(y1,)] += 1

    total = float(n - start)
    te = 0.0
    for (yt, y1, xl), n_yyx in c_y_y1_x.items():
        p_yyx = n_yyx / total
        p_y_given_y1x = n_yyx / c_y1_x[(y1, xl)]
        p_y_given_y1 = c_y_y1[(yt, y1)] / c_y1[(y1,)]
        if p_y_given_y1x > 0 and p_y_given_y1 > 0:
            te += p_yyx * np.log(p_y_given_y1x / p_y_given_y1)
    return float(te)


def ete_with_surrogates(x_sym: np.ndarray, y_sym: np.ndarray, lag: int, m: int, rng: np.random.Generator):
    te = symbolic_te(x_sym, y_sym, lag)
    if m <= 0:
        return te, te, 0.0

    tes = np.empty(m, dtype=float)
    base = x_sym.copy()
    for i in range(m):
        x_shuf = base.copy()
        rng.shuffle(x_shuf)
        tes[i] = symbolic_te(x_shuf, y_sym, lag)

    mu = float(tes.mean())
    sd = float(tes.std(ddof=1)) if m > 1 else 0.0
    ete = te - mu
    z = (te - mu) / (sd + 1e-12)
    return float(ete), float(te), float(z)


def make_asofs(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = pd.Series(index, index=index)
    # prende l'ultimo bar disponibile per ogni giorno
    last_per_day = s.groupby(index.floor("D")).max().tolist()
    return sorted(set(last_per_day))


def window_positions(sym_index: pd.DatetimeIndex, asofs: list[pd.Timestamp], bars: int) -> list[tuple[int, int, pd.Timestamp]]:
    # ritorna (lo, hi, asof) su sym_index, dove slice = [lo:hi] include hi-1
    out = []
    for t_end in asofs:
        start_time = t_end - pd.Timedelta(minutes=BAR_MINUTES * (bars - 1))
        lo = sym_index.searchsorted(start_time, side="left")
        hi = sym_index.searchsorted(t_end, side="right")
        if hi - lo >= 10:
            out.append((lo, hi, t_end))
    return out


def compute_pair(L: str, F: str,
                 sym: dict[str, np.ndarray],
                 sym_index: pd.DatetimeIndex,
                 pos_list: list[tuple[int,int,pd.Timestamp]],
                 surrogates: int = SURROGATES) -> list[dict]:

    rng = np.random.default_rng(abs(hash((L, F, RNG_SEED))) % (2**32))
    x_all = sym[L]
    y_all = sym[F]

    daily = []
    for lo, hi, t_end in pos_list:
        xw = x_all[lo:hi]
        yw = y_all[lo:hi]
        best = None
        for lag in LAGS:
            ete, te, z = ete_with_surrogates(xw, yw, lag, surrogates, rng)
            if best is None or ete > best["ete"]:
                best = {"ete": ete, "te": te, "z": z, "lag": lag}
        sig = bool(best["z"] >= Z_TH and best["ete"] > 0)
        daily.append((t_end, int(best["lag"]), float(best["ete"]), float(best["te"]), sig))

    if len(daily) < LONG_WINDOW_DAYS:
        return []

    # rolling 21 giorni sui daily
    rows = []
    asof_arr = [d[0] for d in daily]
    lag_arr = np.array([d[1] for d in daily], dtype=int)
    ete_arr = np.array([d[2] for d in daily], dtype=float)
    te_arr = np.array([d[3] for d in daily], dtype=float)
    sig_arr = np.array([d[4] for d in daily], dtype=float)

    for i in range(LONG_WINDOW_DAYS - 1, len(daily)):
        sl = slice(i - LONG_WINDOW_DAYS + 1, i + 1)
        ete_w = ete_arr[sl]
        te_w = te_arr[sl]
        sig_w = sig_arr[sl]
        lag_w = lag_arr[sl]

        # moda lag
        mode = int(pd.Series(lag_w).mode().iloc[0])

        rows.append({
            "leader": L,
            "follower": F,
            "ete": float(ete_w.mean()),
            "ste": float(te_w.mean()),
            "best_lag": mode,
            "ete_hit_rate": float(sig_w.mean()),
            "ete_std": float(ete_w.std(ddof=1)) if len(ete_w) > 1 else 0.0,
            "asof": asof_arr[i],
        })
    return rows


def run():
    wide = pd.read_parquet(RET_PATH).sort_index()
    wide.index = pd.to_datetime(wide.index, utc=True)

    # simboli su tutta la serie
    span = (M - 1) * TAU
    sym = {a: ordinal_symbols(wide[a].astype(float).values) for a in wide.columns}
    sym_index = wide.index[span:]

    asofs = make_asofs(wide.index)
    pos_list = window_positions(sym_index, asofs, SHORT_BARS)

    followers = [c for c in wide.columns if c not in LEADERS]

    tasks = []
    for L in LEADERS:
        for F in wide.columns:
            if F == L:
                continue
            tasks.append((L, F))

    n_jobs = min(24, os.cpu_count() or 1)
    print(f"[INFO] tasks={len(tasks)}  n_jobs={n_jobs}  surrogates={SURROGATES}")

    results = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1)(
        delayed(compute_pair)(L, F, sym, sym_index, pos_list, SURROGATES)
        for (L, F) in tasks
    )

    rows = [r for sub in results for r in sub]
    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise RuntimeError("Nessun risultato: controlla dati/parametri.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"[OK] causal_lookup -> {OUT_PATH} rows={len(out):,}")


if __name__ == "__main__":
    run()
