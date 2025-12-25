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
TIMEFRAME = '1h'
MIN_CORRELATION = 0.4
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
            try:
                symbol = os.path.basename(f).replace('.csv', '').split('_')[0].upper()
                df = pd.read_csv(f)
                
                # Normalizzazione nomi colonne
                df.columns = [c.lower() for c in df.columns]
                
                # Identificazione colonne chiave
                date_col = next((c for c in df.columns if 'date' in c or 'time' in c or 'timestamp' in c), None)
                close_col = next((c for c in df.columns if 'close' in c), None)
                
                if date_col and close_col:
                    # FIX CRITICO TIMESTAMP: Rilevamento automatico unità
                    first_val = df[date_col].iloc[0]
                    
                    # Se è numerico (non stringa)
                    if isinstance(first_val, (int, float, np.number)):
                        # Se il numero è molto grande (> 10^11), è in millisecondi (es. Binance)
                        # Se è interpretato come nanosecondi da pandas, da il 1970.
                        if first_val > 10**11: 
                            df[date_col] = pd.to_datetime(df[date_col], unit='ms')
                        else:
                            df[date_col] = pd.to_datetime(df[date_col], unit='s')
                    else:
                        # Parsing stringa standard
                        df[date_col] = pd.to_datetime(df[date_col])

                    df = df.set_index(date_col).sort_index()
                    df = df[~df.index.duplicated(keep='first')]
                    data_frames[symbol] = df[close_col]
            except Exception as e:
                print(f"[!] Errore caricamento {f}: {e}")

        if not data_frames:
            print("[!] Nessun dato caricato.")
            return

        # Unisci tutto
        self.prices = pd.concat(data_frames, axis=1).dropna()
        self.assets = self.prices.columns.tolist()
        
        # Calcolo Log Returns
        self.returns = np.log(self.prices / self.prices.shift(1)).dropna()
        
        print(f"[+] Dati allineati corretti. Asset: {len(self.assets)}. Righe: {len(self.returns)}")
        if not self.returns.empty:
            print(f"    Range Reale: {self.returns.index[0]} - {self.returns.index[-1]}")

    def analyze_lead_lag_relaxed(self):
        """
        Analisi Lead-Lag con Cross-Correlazione.
        """
        print("[*] Avvio analisi Lead-Lag (Metodo CCF)...")
        pairs_found = []

        if self.returns.empty:
            print("[!] DataFrame vuoto, impossibile analizzare.")
            return

        corr_matrix = self.returns.corr()
        
        for leader in self.assets:
            for follower in self.assets:
                if leader == follower:
                    continue
                
                # Filtro preliminare correlazione base
                if abs(corr_matrix.loc[leader, follower]) < MIN_CORRELATION:
                    continue

                series_leader = self.returns[leader]
                series_follower = self.returns[follower]
                
                base_corr = series_leader.corr(series_follower)
                best_lag = 0
                max_corr = base_corr

                # Testiamo Lag da 1 a MAX_LAG
                for lag in range(1, MAX_LAG + 1):
                    lagged_corr = series_leader.shift(lag).corr(series_follower)
                    if lagged_corr > max_corr:
                        max_corr = lagged_corr
                        best_lag = lag
                
                # Criteri: Deve migliorare la correlazione e il miglior lag deve essere > 0
                # Soglia di miglioramento (0.01) resa meno stringente per catturare micro-inefficienze
                if best_lag > 0 and (max_corr - base_corr) > 0.01:
                    pairs_found.append({
                        'Leader': leader,
                        'Follower': follower,
                        'Lag': best_lag,
                        'Base_Corr': round(base_corr, 4),
                        'Lagged_Corr': round(max_corr, 4),
                        'Strength': round(max_corr - base_corr, 4)
                    })

        results = pd.DataFrame(pairs_found)
        if not results.empty:
            results = results.sort_values(by='Strength', ascending=False)
            print(f"[+] Trovate {len(results)} coppie Lead-Lag.")
            print(results.head(15).to_string(index=False))
            filename = f'report_leadlag_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            results.to_csv(filename, index=False)
            print(f"[OUTPUT] Salvato: {filename}")
        else:
            print("[INFO] Nessuna coppia Lead-Lag significativa trovata.")

    def analyze_seesaw_index(self):
        print("[*] Analisi Effetto Seesaw (Index Based)...")
        avail_large = [a for a in LARGE_CAPS if a in self.assets]
        avail_small = [a for a in self.assets if a not in avail_large]

        if not avail_large or not avail_small:
            return

        large_idx = self.returns[avail_large].mean(axis=1)
        small_idx = self.returns[avail_small].mean(axis=1)

        rolling_corr = large_idx.rolling(window=30).corr(small_idx)
        avg_corr = rolling_corr.mean()
        min_corr = rolling_corr.min()

        print(f"    Large Caps: {len(avail_large)} | Small Caps: {len(avail_small)}")
        print(f"    Correlazione Media: {avg_corr:.3f}")
        print(f"    Momento Max Seesaw (Divergenza): {min_corr:.3f}")

        if min_corr < 0:
            print("[+] Trovata divergenza negativa (Seesaw potenziale) nel periodo.")
        else:
            print("[-] Nessuna divergenza significativa rilevata.")

    def run_clustering_viz(self, n_clusters=4):
        print("[*] Generazione Clustering e Heatmap...")
        corr = self.returns.corr()
        
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
        affinity_matrix = (corr + 1) / 2
        labels = sc.fit_predict(affinity_matrix)
        
        clustered_series = pd.Series(labels, index=self.assets, name='Cluster')
        df_clusters = clustered_series.sort_values().to_frame()
        
        print("\n=== CLUSTER IDENTIFICATI ===")
        for i in range(n_clusters):
            members = df_clusters[df_clusters['Cluster'] == i].index.tolist()
            print(f"Cluster {i}: {', '.join(members)}")
            
        plt.figure(figsize=(12, 10))
        sorted_assets = df_clusters.index
        sns.heatmap(self.returns[sorted_assets].corr(), cmap='coolwarm', center=0, annot=False)
        plt.title('Correlation Heatmap (Clustered)')
        plt.tight_layout()
        plt.savefig(f'heatmap_clusters_{datetime.now().strftime("%Y%m%d")}.png')
        print("[OUTPUT] Heatmap salvata.")

if __name__ == "__main__":
    engine = CryptoAlphaEngine(DATA_DIR)
    engine.load_data()
    
    if not engine.prices.empty:
        engine.analyze_lead_lag_relaxed()
        engine.analyze_seesaw_index()
        engine.run_clustering_viz()
