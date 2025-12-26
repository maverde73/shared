from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

RET_PATH = Path("data/graphs/returns_wide.parquet")
OUT_PATH = Path("data/graphs/causal_lookup.parquet")

# === PARAMETRI ===
LEADERS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]

BAR_MINUTES = 15
LAGS = list(range(1, 13))  # 1..12 barre => 15m..3h

SHORT_WINDOW_HOURS = 6     # TE "istantanea" su 6h
SHORT_BARS = int((SHORT_WINDOW_HOURS * 60) / BAR_MINUTES)  # 24 barre

LONG_WINDOW_DAYS = 21      # stabilità su 21 giorni
SURROGATES = 30            # shuffled X (ETE); riduci se troppo lento
Z_TH = 2.0                 # significatività: z-score contro surrogati

# Symbolic embedding (ordinal patterns)
M = 3      # dimensione embedding (3 ok per intraday)
TAU = 1    # ritardo interno (1 barra)

# Snapshot giornaliero: fine giornata UTC (puoi cambiare)
ASOF_TIME = "23:45"  # ultimo bar del giorno su 15m


# ------------------------------
# Utility: ordinal patterns -> simboli
# ------------------------------
def ordinal_symbols(x: np.ndarray, m: int = M, tau: int = TAU) -> np.ndarray:
    """
    Converte una serie 1D in simboli ordinali (permutazioni) di dimensione m.
    Ritorna array di int (id simbolo). Gli ultimi m-1*tau campioni non producono simbolo.
    """
    n = len(x)
    span = (m - 1) * tau
    if n <= span:
        return np.array([], dtype=np.int32)

    # Costruiamo vettori [x[t-span], ..., x[t]] con passo tau
    # simbolo al tempo t (indice t) => pattern dei m valori
    syms = []
    for t in range(span, n):
        window = x[t - span : t + 1 : tau]
        # ordinal pattern: ranking (argsort) -> tuple
        # tie-breaking stabile: aggiungiamo un jitter minimo deterministico
        # per evitare pattern instabili con valori identici
        jitter = (np.arange(m) * 1e-12)
        order = tuple(np.argsort(window + jitter))
        syms.append(order)

    # Mappa permutazioni a id
    # m=3 => max 6 simboli, m=4 => 24, ecc.
    uniq = {p: i for i, p in enumerate(sorted(set(syms)))}
    return np.array([uniq[p] for p in syms], dtype=np.int32)


# ------------------------------
# STE: Transfer Entropy su simboli discreti
# TE(X->Y) = sum p(y_t, y_{t-1}, x_{t-L}) log p(y_t|y_{t-1},x)/p(y_t|y_{t-1})
# ------------------------------
def symbolic_te(x_sym: np.ndarray, y_sym: np.ndarray, lag: int) -> float:
    """
    x_sym, y_sym: simboli ordinali già calcolati (allineati temporalmente sui returns)
    lag: quante barre X precede Y.
    """
    # Allineamento:
    # y_t usa indice i
    # y_{t-1} indice i-1
    # x_{t-lag} indice i-lag
    # Quindi i deve essere >= max(1, lag)
    n = min(len(x_sym), len(y_sym))
    if n <= max(1, lag) + 1:
        return 0.0

    start = max(1, lag)
    # conteggi
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
    if total <= 0:
        return 0.0

    te = 0.0
    # TE = sum p(yt,y1,x) * log( p(yt|y1,x) / p(yt|y1) )
    for (yt, y1, xl), n_yyx in c_y_y1_x.items():
        p_yyx = n_yyx / total
        p_y_given_y1x = n_yyx / c_y1_x[(y1, xl)]
        p_y_given_y1 = c_y_y1[(yt, y1)] / c_y1[(y1,)]
        # guardie numeriche
        if p_y_given_y1x > 0 and p_y_given_y1 > 0:
            te += p_yyx * np.log(p_y_given_y1x / p_y_given_y1)

    return float(te)


def ete_with_surrogates(x_sym: np.ndarray, y_sym: np.ndarray, lag: int, m: int = SURROGATES) -> tuple[float, float, float]:
    """
    Ritorna: (ete, te, zscore)
    zscore = (te - mean_surr)/std_surr
    """
    te = symbolic_te(x_sym, y_sym, lag)
    if m <= 0:
        return (te, te, 0.0)

    rng = np.random.default_rng(42)
    tes = []
    for _ in range(m):
        x_shuf = x_sym.copy()
        rng.shuffle(x_shuf)  # distrugge struttura temporale
        tes.append(symbolic_te(x_shuf, y_sym, lag))

    tes = np.array(tes, dtype=float)
    mu = float(tes.mean())
    sd = float(tes.std(ddof=1)) if len(tes) > 1 else 0.0
    ete = te - mu
    z = (te - mu) / (sd + 1e-12)
    return (float(ete), float(te), float(z))


# ------------------------------
# Main: calcolo giornaliero + rolling stabilità
# ------------------------------
def run():
    wide = pd.read_parquet(RET_PATH).sort_index()
    wide.index = pd.to_datetime(wide.index, utc=True)

    # sanity leaders presenti
    for L in LEADERS:
        if L not in wide.columns:
            raise ValueError(f"Leader mancante in returns_wide: {L}")

    # snapshot giornalieri: prendiamo l'ultimo bar del giorno (23:45)
    # costruendo una lista di timestamp esistenti più vicini
    days = wide.index.floor("D").unique()

    asofs = []
    for d in days:
        target = pd.Timestamp(f"{d.date()} {ASOF_TIME}", tz="UTC")
        # se non esiste esattamente, prendi il bar precedente disponibile
        idx = wide.index[wide.index <= target]
        if len(idx) == 0:
            continue
        asofs.append(idx[-1])

    asofs = sorted(set(asofs))
    if len(asofs) < LONG_WINDOW_DAYS + 5:
        raise ValueError("Pochi snapshot giornalieri: controlla indice/ASOF_TIME.")

    # memorizziamo per ogni (leader,follower) la serie giornaliera dei risultati "istantanei"
    daily = defaultdict(list)  # key=(L,F) -> list di dict (asof, best_lag, ete, z, significant)

    # Pre-calcolo simboli ordinali su tutta la serie (poi facciamo slicing per finestra)
    # Nota: i simboli accorciano la serie di span=(M-1)*TAU
    span = (M - 1) * TAU
    sym = {}
    for a in wide.columns:
        x = wide[a].astype(float).values
        sym[a] = ordinal_symbols(x, m=M, tau=TAU)

    # Mappa time index per simboli:
    # sym[a][k] corrisponde a wide.index[k+span]
    sym_index = wide.index[span:]

    # funzione per estrarre finestre simboliche allineate
    def slice_syms(asset: str, t_end: pd.Timestamp, bars: int) -> np.ndarray:
        # finestra su returns: ultimi 'bars' fino a t_end inclusivo
        # convertiamo in range su sym_index
        # start_time = t_end - (bars-1)*15m
        start_time = t_end - pd.Timedelta(minutes=BAR_MINUTES * (bars - 1))
        mask = (sym_index >= start_time) & (sym_index <= t_end)
        return sym[asset][mask]

    # CALCOLO giornaliero "istantaneo" (6h) per ogni asof
    for t_end in asofs:
        for L in LEADERS:
            xw = slice_syms(L, t_end, SHORT_BARS)
            if len(xw) < 10:
                continue

            for F in wide.columns:
                if F == L:
                    continue
                yw = slice_syms(F, t_end, SHORT_BARS)
                if len(yw) < 10:
                    continue

                best = None
                for lag in LAGS:
                    ete, te, z = ete_with_surrogates(xw, yw, lag, m=SURROGATES)
                    if best is None or ete > best["ete"]:
                        best = {"ete": ete, "te": te, "z": z, "lag": lag}

                sig = bool(best["z"] >= Z_TH and best["ete"] > 0)
                daily[(L, F)].append({
                    "asof": t_end,
                    "best_lag": int(best["lag"]),
                    "ete_inst": float(best["ete"]),
                    "te_inst": float(best["te"]),
                    "z_inst": float(best["z"]),
                    "significant": sig,
                })

        print(f"[DAY] computed {t_end}")

    # ROLLING 21g: costruiamo causal_lookup giornaliero
    rows = []
    for (L, F), lst in daily.items():
        df = pd.DataFrame(lst).sort_values("asof")
        if len(df) < LONG_WINDOW_DAYS:
            continue

        # rolling window su ultimi 21 record (giorni disponibili)
        for i in range(LONG_WINDOW_DAYS - 1, len(df)):
            w = df.iloc[i - LONG_WINDOW_DAYS + 1 : i + 1]

            ete_mean = float(w["ete_inst"].mean())
            ete_std = float(w["ete_inst"].std(ddof=1)) if len(w) > 1 else 0.0
            hit = float(w["significant"].mean())

            # lag moda
            lag_mode = int(w["best_lag"].mode().iloc[0])

            # valore ste riportato: qui mettiamo TE medio (non “effective”)
            ste_mean = float(w["te_inst"].mean())

            rows.append({
                "leader": L,
                "follower": F,
                "ete": ete_mean,
                "ste": ste_mean,
                "best_lag": lag_mode,
                "ete_hit_rate": hit,
                "ete_std": ete_std,
                "asof": w["asof"].iloc[-1],
            })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise RuntimeError("Nessuna riga generata: controlla dati/parametri.")

    # Salva
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"[OK] causal_lookup -> {OUT_PATH} rows={len(out):,}")


if __name__ == "__main__":
    run()
