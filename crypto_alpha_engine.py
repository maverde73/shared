import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import correlate
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import warnings
from datetime import datetime

# Ignora warning minori per pulizia output
warnings.filterwarnings('ignore')

# Configurazione Hardware (Ottimizzata per Ryzen 5900X)
N_JOBS = 20  # Lasciamo 4 thread liberi per il sistema operativo
VERBOSITY = 1

class CryptoDataLoader:
    """
    Gestisce il caricamento, la pulizia e l'allineamento dei dati CSV.
    Assume il formato: timestamp,open,high,low,close,volume
    """
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.tickers = []
        self.price_data = pd.DataFrame()
        self.volume_data = pd.DataFrame()

    def load_data(self):
        print(f"[*] Caricamento dati da {self.data_folder}...")
        all_files = glob.glob(os.path.join(self.data_folder, "*.csv"))
        
        if not all_files:
            raise FileNotFoundError("Nessun file CSV trovato nella cartella specificata.")

        # Funzione helper per caricamento parallelo
        def process_file(filepath):
            try:
                # Estrae il simbolo dal nome file (es. ADAUSDT_5m.csv -> ADAUSDT)
                filename = os.path.basename(filepath)
                ticker = filename.split('_')[0]
                
                df = pd.read_csv(filepath)
                # Conversione Timestamp (ms -> datetime)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Rimuove duplicati
                df = df[~df.index.duplicated(keep='first')]
                
                return ticker, df['close'], df['volume']
            except Exception as e:
                print(f"[!] Errore nel file {filepath}: {e}")
                return None

        # Esecuzione parallela
        results = Parallel(n_jobs=N_JOBS)(delayed(process_file)(f) for f in all_files)
        
        # Assemblaggio DataFrame
        prices = {}
        volumes = {}
        
        for res in results:
            if res:
                ticker, close_series, vol_series = res
                prices[ticker] = close_series
                volumes[ticker] = vol_series
        
        self.price_data = pd.DataFrame(prices)
        self.volume_data = pd.DataFrame(volumes)
        
        # Forward Fill per buchi dati (fino a un limite ragionevole), poi drop
        self.price_data.ffill(limit=12, inplace=True) # Max 1 ora di buco su 5m
        self.price_data.dropna(axis=0, how='any', inplace=True) # Rimuove righe dove manca qualche crypto
        
        # Allineamento volumi
        self.volume_data = self.volume_data.reindex(self.price_data.index).fillna(0)
        
        self.tickers = self.price_data.columns.tolist()
        print(f"[+] Dati caricati e allineati. Totale Asset: {len(self.tickers)}. Range temporale: {self.price_data.index.min()} - {self.price_data.index.max()}")
        return self.price_data, self.volume_data

    def get_log_returns(self):
        """Calcola i rendimenti logaritmici necessari per la stazionarietà."""
        return np.log(self.price_data / self.price_data.shift(1)).dropna()

class AlphaEngine:
    """
    Core analitico per trovare correlazioni, Lead-Lag e Clustering.
    """
    def __init__(self, price_df, volume_df, log_returns):
        self.prices = price_df
        self.volumes = volume_df
        self.returns = log_returns
        self.assets = price_df.columns.tolist()

    def analyze_lead_lag(self, window_size=60, max_lag=12, correlation_threshold=0.65, stability_threshold=0.7):
        """
        Analisi Rolling Cross-Correlation per identificare coppie Lead-Lag stabili.
        
        Params:
            window_size: Finestra rolling (es. 60 periodi da 5m = 5 ore)
            max_lag: Massimo ritardo da testare (es. 12 periodi = 1 ora)
            correlation_threshold: Soglia minima di correlazione per considerare un segnale
            stability_threshold: Percentuale di tempo in cui la correlazione deve reggere
        """
        print("[*] Avvio analisi Lead-Lag (Heavy CPU Task)...")
        
        # Generiamo tutte le coppie possibili (escludendo auto-correlazioni)
        import itertools
        pairs = list(itertools.permutations(self.assets, 2))
        print(f"[*] Analisi di {len(pairs)} coppie potenziali.")

        def evaluate_pair(leader, laggard):
            # Ottimizzazione: Usiamo numpy arrays per velocità
            y_leader = self.returns[leader].values
            y_laggard = self.returns[laggard].values
            
            # Matrice per salvare le correlazioni rolling per ogni lag
            # Shape: (Time, Lags)
            # Nota: Pandas rolling corr è lento. Usiamo un approccio vettorializzato se possibile o semplificato.
            # Per precisione "senza compromessi", usiamo pandas rolling ma parallelizzato sulle coppie.
            
            # Creiamo un DF temporaneo per la coppia
            pair_df = pd.DataFrame({
                'lead': self.returns[leader],
                'lag': self.returns[laggard]
            })
            
            # Calcoliamo la correlazione massima e il lag ottimale nel tempo
            # Questo è computazionalmente costoso, lo facciamo su finestre
            
            best_lags = []
            max_corrs = []
            
            # Iteriamo con step per ridurre carico (es. ricalcoliamo ogni 12 barre / 1 ora)
            step = 12 
            
            for i in range(window_size, len(pair_df), step):
                window_lead = y_leader[i-window_size : i]
                window_lag = y_laggard[i-window_size : i]
                
                # Cross-correlazione via FFT (scipy)
                # mode='full' restituisce array lungo 2*n-1. Il centro è lag 0.
                cc = correlate(window_lead - np.mean(window_lead), 
                               window_lag - np.mean(window_lag), mode='full') / (window_size * np.std(window_lead) * np.std(window_lag))
                
                lags = np.arange(-window_size + 1, window_size)
                
                # Filtriamo solo i lag positivi (Leader deve anticipare Laggard)
                # In correlate(in1, in2), se in1 anticipa in2, il picco è su lag negativi o positivi in base alla convenzione.
                # Qui usiamo la convenzione: se Leader anticipa, cerchiamo picco dove shiftiamo Laggard "indietro" per matchare.
                
                valid_mask = (lags > 0) & (lags <= max_lag)
                valid_lags = lags[valid_mask]
                valid_cc = cc[valid_mask]
                
                if len(valid_cc) == 0:
                    max_corrs.append(0)
                    best_lags.append(0)
                    continue

                idx_max = np.argmax(np.abs(valid_cc))
                max_corr = valid_cc[idx_max]
                best_lag = valid_lags[idx_max]
                
                max_corrs.append(max_corr)
                best_lags.append(best_lag)

            # Analisi Stabilità
            max_corrs = np.array(max_corrs)
            
            # Percentuale di tempo in cui la correlazione è sopra la soglia
            stability_score = np.sum(max_corrs > correlation_threshold) / len(max_corrs) if len(max_corrs) > 0 else 0
            
            if stability_score >= stability_threshold:
                # Calcola il Lag Modale (il ritardo più frequente)
                import statistics
                try:
                    modal_lag = statistics.mode([l for l, c in zip(best_lags, max_corrs) if c > correlation_threshold])
                except:
                    modal_lag = best_lags[0]
                    
                return {
                    'Leader': leader,
                    'Laggard': laggard,
                    'Stability': round(stability_score, 2),
                    'Avg_Correlation': round(np.mean(max_corrs[max_corrs > correlation_threshold]), 2),
                    'Optimal_Lag_Steps': modal_lag,
                    'Optimal_Lag_Minutes': modal_lag * 5 # Assumendo candele 5m
                }
            return None

        # Esecuzione parallela
        results = Parallel(n_jobs=N_JOBS)(delayed(evaluate_pair)(p[0], p[1]) for p in pairs)
        
        # Filtra None
        valid_pairs = [r for r in results if r is not None]
        valid_pairs.sort(key=lambda x: x['Stability'], reverse=True)
        
        print(f"[+] Trovate {len(valid_pairs)} coppie stabili.")
        return pd.DataFrame(valid_pairs)

    def analyze_seesaw_effect(self, lookback_window=288):
        """
        Analizza l'effetto Seesaw (Large Cap vs Small Cap).
        Usa Volume * Prezzo come proxy per la Capitalizzazione/Importanza.
        
        lookback_window: 288 periodi (24 ore su 5m) per definire chi è "Large" oggi.
        """
        print("[*] Analisi Effetto Seesaw (Large vs Small)...")
        
        # 1. Determina Size Factor (Volume Medio * Prezzo Medio nella finestra)
        recent_prices = self.prices.iloc[-lookback_window:]
        recent_vols = self.volumes.iloc[-lookback_window:]
        
        # Proxy Liquidity
        liquidity_proxy = (recent_prices * recent_vols).mean()
        
        # Classificazione (Top 20% vs Bottom 40%)
        threshold_large = liquidity_proxy.quantile(0.8)
        threshold_small = liquidity_proxy.quantile(0.4)
        
        large_caps = liquidity_proxy[liquidity_proxy >= threshold_large].index.tolist()
        small_caps = liquidity_proxy[liquidity_proxy <= threshold_small].index.tolist()
        
        print(f"[*] Identificate {len(large_caps)} Large Caps e {len(small_caps)} Small Caps.")
        
        seesaw_pairs = []
        
        # Testiamo se i ritorni delle Large Cap predicono NEGATIVAMENTE le Small Cap
        # Usiamo un lag fisso di 1 periodo (5m) per semplicità, ma iteriamo su tutte le combinazioni
        
        # Creiamo un indice sintetico "Large Cap Index" (media dei ritorni)
        large_cap_index = self.returns[large_caps].mean(axis=1)
        
        for small in small_caps:
            # Calcolo correlazione tra Index Large(t-1) e Small(t)
            # Shiftiamo l'indice large in avanti per allinearlo al tempo t dello small
            predictor = large_cap_index.shift(1)
            target = self.returns[small]
            
            # Correlazione su tutta la storia disponibile (si può fare rolling per raffinatezza)
            correlation = predictor.corr(target)
            
            # Se la correlazione è significativamente negativa
            if correlation < -0.2: # Soglia empirica per il Seesaw
                seesaw_pairs.append({
                    'Type': 'Seesaw',
                    'Predictor': 'LargeCap_Index',
                    'Target': small,
                    'Correlation': round(correlation, 3),
                    'Note': 'Negative Cross-Predictability'
                })
                
        print(f"[+] Trovate {len(seesaw_pairs)} relazioni Seesaw significative.")
        return pd.DataFrame(seesaw_pairs)

    def perform_spectral_clustering(self, n_clusters=5):
        """
        Raggruppa gli asset in cluster basati sulla similarità dei movimenti.
        Identifica il "Volume Leader" per ogni cluster.
        """
        print("[*] Esecuzione Spectral Clustering...")
        
        # 1. Calcolo matrice di correlazione (Affinity Matrix)
        corr_matrix = self.returns.corr()
        
        # Convertiamo correlazione in distanza/affinità positiva (0 a 1)
        # (1 + corr) / 2 -> mappa -1 a 0, e 1 a 1.
        affinity = (1 + corr_matrix) / 2
        affinity = affinity.fillna(0) # Safety
        
        # 2. Clustering
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
        labels = sc.fit_predict(affinity)
        
        clusters = {}
        
        # 3. Analisi per Cluster
        results = []
        for cluster_id in range(n_clusters):
            members = corr_matrix.index[labels == cluster_id].tolist()
            
            if not members:
                continue
                
            # Trova il Leader del Cluster (Highest Average Volume)
            avg_vols = self.volumes[members].mean()
            leader = avg_vols.idxmax()
            
            results.append({
                'Cluster_ID': cluster_id,
                'Members_Count': len(members),
                'Leader_Asset': leader,
                'Members': ", ".join(members)
            })
            
        print(f"[+] Clustering completato. {n_clusters} regimi identificati.")
        return pd.DataFrame(results)

def main():
    print("===================================================")
    print("   CRYPTO ALPHA ENGINE - FEDORA/RYZEN OPTIMIZED    ")
    print("===================================================")
    
    # PATH DATI (Modificare con il percorso reale su Fedora)
    DATA_PATH = "./data_csv" 
    
    # 1. Creazione cartella dati se non esiste (per test)
    if not os.path.exists(DATA_PATH):
        print(f"[!] Cartella {DATA_PATH} non trovata. Creare la cartella e inserire i file CSV (es. ADAUSDT_5m.csv).")
        return

    # 2. Inizializzazione e Caricamento
    loader = CryptoDataLoader(DATA_PATH)
    try:
        prices, volumes = loader.load_data()
    except Exception as e:
        print(e)
        return

    if prices.empty:
        print("[!] Nessun dato caricato. Uscita.")
        return

    # 3. Preprocessing
    log_returns = loader.get_log_returns()
    
    # 4. Analisi
    engine = AlphaEngine(prices, volumes, log_returns)
    
    # A. Lead-Lag
    lead_lag_df = engine.analyze_lead_lag(
        window_size=120,    # 10 Ore di lookback per la rolling
        max_lag=6,          # Max 30 min di ritardo
        correlation_threshold=0.60, 
        stability_threshold=0.75
    )
    
    # B. Seesaw Effect
    seesaw_df = engine.analyze_seesaw_effect()
    
    # C. Clustering
    cluster_df = engine.perform_spectral_clustering(n_clusters=4)
    
    # 5. Salvataggio Report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not lead_lag_df.empty:
        lead_lag_df.to_csv(f"report_lead_lag_{timestamp}.csv", index=False)
        print(f"[OUTPUT] Lead-Lag report salvato: report_lead_lag_{timestamp}.csv")
        print("\n--- TOP 5 STABLE PAIRS ---")
        print(lead_lag_df.head(5))
    else:
        print("[INFO] Nessuna coppia Lead-Lag stabile trovata con i parametri attuali.")
        
    if not seesaw_df.empty:
        seesaw_df.to_csv(f"report_seesaw_{timestamp}.csv", index=False)
        print(f"[OUTPUT] Seesaw report salvato: report_seesaw_{timestamp}.csv")
    
    cluster_df.to_csv(f"report_clusters_{timestamp}.csv", index=False)
    print(f"[OUTPUT] Cluster report salvato: report_clusters_{timestamp}.csv")

if __name__ == "__main__":
    main()
