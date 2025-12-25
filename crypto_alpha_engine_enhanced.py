import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# --- CONFIGURAZIONE ---
DATA_DIR = './data_csv'
TIMEFRAME = '1h'  # Assunto basato sull'output precedente, adattabile
MIN_CORRELATION = 0.4  # Abbassato da 0.7/0.8 standard
MAX_LAG = 3  # Guardiamo indietro di 3 periodi (es. 3 ore)
LARGE_CAPS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']

class CryptoAlphaEngine:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.prices = pd.DataFrame()
        self.returns = pd.DataFrame()
        self.assets = []

    def load_data(self):
        print(f"[*] Caricamento dati da {self.data_dir}...")
        all_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        
        data_frames = {}
        for f in all_files:
            # Assume formato file: Ticker_Timeframe.csv o simile, o interno al CSV
            try:
                # Tenta di estrarre il simbolo dal nome file
                symbol = os.path.basename(f).replace('.csv', '').split('_')[0].upper()
                
                # Leggi CSV (Assume colonne standard: timestamp/date, open, high, low, close, volume)
                df = pd.read_csv(f)
                
                # Normalizzazione nomi colonne
                df.columns = [c.lower() for c in df.columns]
                date_col = next((c for c in df.columns if 'date' in c or 'time' in c), None)
                close_col = next((c for c in df.columns if 'close' in c), None)
                
                if date_col and close_col:
                    df[date_col] = pd.to_datetime(df[date_col])
                    df = df.set_index(date_col).sort_index()
                    # Rimuovi duplicati indice
                    df = df[~df.index.duplicated(keep='first')]
                    data_frames[symbol] = df[close_col]
            except Exception as e:
                print(f"[!] Errore caricamento {f}: {e}")

        if not data_frames:
            print("[!] Nessun dato caricato.")
            return

        # Unisci tutto in un unico DataFrame allineato temporalmente
        self.prices = pd.concat(data_frames, axis=1).dropna()
        self.assets = self.prices.columns.tolist()
        
        # Calcolo Log Returns
        self.returns = np.log(self.prices / self.prices.shift(1)).dropna()
        
        print(f"[+] Dati allineati. Asset: {len(self.assets)}. Righe: {len(self.returns)}")
        print(f"    Range: {self.returns.index[0]} - {self.returns.index[-1]}")

    def analyze_lead_lag_relaxed(self):
        """
        Usa la Cross-Correlazione (CCF) per trovare Lead-Lag.
        Se corr(Leader(t-lag), Follower(t)) > corr(Leader(t), Follower(t)), c'è un lead.
        """
        print("[*] Avvio analisi Lead-Lag (Metodo CCF)...")
        pairs_found = []

        corr_matrix = self.returns.corr()
        
        # Itera su tutte le coppie
        for leader in self.assets:
            for follower in self.assets:
                if leader == follower:
                    continue
                
                # Filtro pre-calcolo: devono essere almeno un po' correlati
                if abs(corr_matrix.loc[leader, follower]) < MIN_CORRELATION:
                    continue

                # Calcola cross-correlazione per lag 1 a MAX_LAG
                series_leader = self.returns[leader]
                series_follower = self.returns[follower]
                
                base_corr = series_leader.corr(series_follower)
                best_lag = 0
                max_corr = base_corr

                for lag in range(1, MAX_LAG + 1):
                    # Shiftiamo il leader IN AVANTI (t-lag) rispetto al follower
                    lagged_corr = series_leader.shift(lag).corr(series_follower)
                    
                    if lagged_corr > max_corr:
                        max_corr = lagged_corr
                        best_lag = lag
                
                # Se la correlazione migliora significativamente con il lag
                if best_lag > 0 and (max_corr - base_corr) > 0.03: # 0.03 è il "Lead Premium" richiesto
                    pairs_found.append({
                        'Leader': leader,
                        'Follower': follower,
                        'Lag': best_lag,
                        'Base_Corr': round(base_corr, 3),
                        'Lagged_Corr': round(max_corr, 3),
                        'Strength': round(max_corr - base_corr, 3)
                    })

        # Ordina per forza del segnale
        results = pd.DataFrame(pairs_found)
        if not results.empty:
            results = results.sort_values(by='Strength', ascending=False)
            print(f"[+] Trovate {len(results)} coppie Lead-Lag potenziali.")
            print(results.head(10).to_string(index=False))
            results.to_csv(f'report_leadlag_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)
        else:
            print("[INFO] Nessuna coppia Lead-Lag trovata con parametri rilassati.")

    def analyze_seesaw_index(self):
        """
        Crea un indice Large Cap vs Small Cap e cerca correlazioni negative.
        """
        print("[*] Analisi Effetto Seesaw (Index Based)...")
        
        # Identifica Large e Small presenti nei dati
        avail_large = [a for a in LARGE_CAPS if a in self.assets]
        avail_small = [a for a in self.assets if a not in avail_large]

        if not avail_large or not avail_small:
            print("[!] Asset insufficienti per Seesaw.")
            return

        # Crea indici equi-pesati
        large_idx = self.returns[avail_large].mean(axis=1)
        small_idx = self.returns[avail_small].mean(axis=1)

        # Correlazione Rolling a 30 periodi
        rolling_corr = large_idx.rolling(window=30).corr(small_idx)
        
        avg_corr = rolling_corr.mean()
        min_corr = rolling_corr.min() # Il punto di massima divergenza

        print(f"    Large Caps: {len(avail_large)} | Small Caps: {len(avail_small)}")
        print(f"    Correlazione Media: {avg_corr:.3f}")
        print(f"    Momento Max Seesaw (Divergenza): {min_corr:.3f}")

        if min_corr < -0.3:
            print("[+] Rilevate fasi di rotazione capitale (Seesaw significativo).")
        else:
            print("[-] Mercato prevalentemente correlato (Risk-On/Risk-Off sincronizzato).")

    def run_clustering_viz(self, n_clusters=4):
        print("[*] Generazione Clustering e Heatmap...")
        corr = self.returns.corr()
        
        # Spectral Clustering
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
        # Trasforma correlazione in affinità (0 a 1, dove -1 diventa 0)
        affinity_matrix = (corr + 1) / 2
        labels = sc.fit_predict(affinity_matrix)
        
        # Organizza output
        clustered_series = pd.Series(labels, index=self.assets, name='Cluster')
        df_clusters = clustered_series.sort_values().to_frame()
        
        print("\n=== CLUSTER IDENTIFICATI (Revisione) ===")
        for i in range(n_clusters):
            members = df_clusters[df_clusters['Cluster'] == i].index.tolist()
            print(f"Cluster {i}: {', '.join(members)}")
            
        # Visualizzazione Matrice di Correlazione Ordinata
        plt.figure(figsize=(12, 10))
        # Ordina la matrice di correlazione in base ai cluster
        sorted_assets = df_clusters.index
        sns.heatmap(self.returns[sorted_assets].corr(), cmap='coolwarm', center=0, annot=False)
        plt.title('Correlation Heatmap (Clustered)')
        plt.tight_layout()
        plt.savefig(f'heatmap_clusters_{datetime.now().strftime("%Y%m%d")}.png')
        print("[OUTPUT] Heatmap salvata come PNG.")

if __name__ == "__main__":
    engine = CryptoAlphaEngine(DATA_DIR)
    engine.load_data()
    
    if not engine.prices.empty:
        engine.analyze_lead_lag_relaxed()
        engine.analyze_seesaw_index()
        engine.run_clustering_viz()
