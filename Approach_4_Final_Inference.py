import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from datetime import datetime

warnings.filterwarnings('ignore')

print("=" * 80)
print("    STARTING Sample Simulation")
print("=" * 80)

# 1. Load Architecture
ART_PATH = r"X:\M.E\Sem1\ADM\ADM\preprocessing\Tier2_finalFailure\Final_ME_Architecture.pkl"
PATH_2018 = r"X:\M.E\Sem1\ADM\ADM\preprocessing\CIC-IDS2018-CSVs\cicids2018_curated_balanced.csv"

art = joblib.load(ART_PATH)
scaler_center = art["scaler"].center_
scaler_scale = art["scaler"].scale_
centroids = art["centroids"]
cluster_stds = art["cluster_stds"]
global_std = art["global_std"]
m_top3 = art["m_top3"]
m_max = art["m_max"]
orig_feature_cols = art["orig_feature_cols"]
keep_indices = art["keep_indices"]
gb_arbiter = art["gb_arbiter"]
t1_cutoff = art["t1_cutoff"]
t2_threshold = art["t2_threshold"]

behavioral_feature_names = [orig_feature_cols[i] for i in keep_indices]

# 2. Loading Blind Live Traffic (Simulating a Network )
TRANSLATION_MAP = {
    'Avg Bwd Segment Size': 'Bwd Seg Size Avg', 'Bwd Packet Length Mean': 'Bwd Pkt Len Mean',
    'Average Packet Size': 'Pkt Size Avg', 'Total Length of Bwd Packets': 'TotLen Bwd Pkts',
    'Subflow Bwd Bytes': 'Subflow Bwd Byts', 'Packet Length Variance': 'Pkt Len Var',
    'Packet Length Std': 'Pkt Len Std', 'Bwd Packet Length Max': 'Bwd Pkt Len Max',
    'Packet Length Mean': 'Pkt Len Mean', 'Max Packet Length': 'Pkt Len Max',
    'Bwd Packet Length Std': 'Bwd Pkt Len Std', 'Fwd IAT Max': 'Fwd IAT Max',
    'Init_Win_bytes_backward': 'Init Bwd Win Byts', 'Fwd Packet Length Max': 'Fwd Pkt Len Max',
    'Subflow Fwd Bytes': 'Subflow Fwd Byts', 'Total Length of Fwd Packets': 'TotLen Fwd Pkts',
    'Fwd IAT Total': 'Fwd IAT Tot', 'Init_Win_bytes_forward': 'Init Fwd Win Byts',
    'Fwd IAT Mean': 'Fwd IAT Mean', 'Fwd IAT Std': 'Fwd IAT Std',
    'Flow IAT Max': 'Flow IAT Max', 'Fwd Packet Length Std': 'Fwd Pkt Len Std',
    'Bwd Header Length': 'Bwd Header Len', 'Flow Duration': 'Flow Duration',
    'Bwd Packets/s': 'Bwd Pkts/s', 'Avg Fwd Segment Size': 'Fwd Seg Size Avg',
    'Fwd Packet Length Mean': 'Fwd Pkt Len Mean', 'Flow IAT Std': 'Flow IAT Std',
    'Bwd IAT Max': 'Bwd IAT Max', 'Fwd Packets/s': 'Fwd Pkts/s',
    'Flow Packets/s': 'Flow Pkts/s', 'Fwd Header Length': 'Fwd Header Len',
    'Fwd Header Length.1': 'Fwd Header Len', 'Flow IAT Mean': 'Flow IAT Mean',
    'Active Mean': 'Active Mean', 'Active Min': 'Active Min', 'Active Max': 'Active Max'
}
cols_18 = [TRANSLATION_MAP[f] for f in orig_feature_cols]

# Load a raw chunk of traffic and immediately dropping all labels to ensure zero data leakage
df_18 = pd.read_csv(PATH_2018, low_memory=False)

# Extract 5 random attacks and 10 random benign flows
is_benign = df_18['Label'].astype(str).str.strip().str.upper() == 'BENIGN'
attacks = df_18[~is_benign].sample(n=5, random_state=42)
benign = df_18[is_benign].sample(n=10, random_state=42)

# Combine and shuffle 
live_traffic = pd.concat([attacks, benign]).sample(frac=1, random_state=42).copy()

# STRIP LABELS
if 'Label' in live_traffic.columns:
    live_traffic.drop(columns=['Label'], inplace=True)
# STRIP LABELS
if 'Label' in live_traffic.columns:
    live_traffic.drop(columns=['Label'], inplace=True)

for col in list(set(cols_18)):
    live_traffic[col] = pd.to_numeric(live_traffic[col], errors='coerce')
live_traffic.fillna(0, inplace=True)

X_live_raw = np.nan_to_num(live_traffic[cols_18].to_numpy(dtype=np.float64), nan=0.0, posinf=1e9, neginf=-1e9)

# 3. Setup Log File and Process Real-Time
LOG_FILE = r"X:\M.E\Sem1\ADM\ADM\IDS_Alerts.log"
print(f" Monitoring blind traffic and writing alerts to: {LOG_FILE}\n")

with open(LOG_FILE, "a", encoding="utf-8") as log:
    log.write(f"\n{'='*75}\n")
    log.write(f" NIDS ENGINE STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.write(f"{'='*75}\n")

    # 4. Process packets one by one
    for i in range(len(X_live_raw)):
        flow_raw = X_live_raw[i:i+1]
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        
        # Core Pipeline Math
        X_scaled = (flow_raw - scaler_center) / scaler_scale
        dists_sq = cdist(X_scaled, centroids, metric='sqeuclidean')
        nr = np.argmin(dists_sq, axis=1)[0]
        min_dist = np.sqrt(dists_sq[0, nr])
        
        eff_s = np.maximum(cluster_stds[nr], np.maximum(0.10 * global_std, 0.05))
        z_mat = np.abs(X_scaled[0] - centroids[nr]) / eff_s
        
        part_z = np.partition(z_mat, -3)[-3:]
        top3_z = np.mean(part_z)
        max_z = np.max(part_z)
        comp_score = (0.60 * (top3_z / m_top3) + 0.40 * (max_z / m_max))
        
        log_entry = ""
        
        # TIER 1 CHECK 
        if comp_score >= t1_cutoff:
            top_3_idx = np.argsort(z_mat)[-3:][::-1]
            
            reasons = ", ".join([f"{orig_feature_cols[idx]} (Dev: {z_mat[idx]:.1f})" for idx in top_3_idx])
            
            log_entry = (
                f"[{timestamp}] [ALERT] TIER 1 WIRE DROP | Flow ID: {i} | Status: MALICIOUS ANOMALY\n"
                f"   -> Severity : Volume Anomaly (Score: {comp_score:.2f} > {t1_cutoff:.2f})\n"
                f"   -> Signature: {reasons}\n\n"
            )
        else:
            # TIER 2 CHECK 
            X_beh = X_scaled[0, keep_indices]
            X_aug = np.hstack([X_beh, min_dist, top3_z, max_z, comp_score]).reshape(1, -1)
            gb_prob = gb_arbiter.predict_proba(X_aug)[0, 1]
            
            if gb_prob >= t2_threshold:
                beh_dev = np.abs(X_beh)
                top_3_idx = np.argsort(beh_dev)[-3:][::-1]
                
                reasons = ", ".join([f"{behavioral_feature_names[idx]} (Dev: {beh_dev[idx]:.1f})" for idx in top_3_idx])
                
                log_entry = (
                    f"[{timestamp}] [ALERT] TIER 2 ARBITER | Flow ID: {i} | Status: MALICIOUS ANOMALY\n"
                    f"   -> Severity : Behavioral Threat (Confidence: {gb_prob*100:.1f}%)\n"
                    f"   -> Signature: {reasons}\n\n"
                )
        
        if log_entry:
            log.write(log_entry)

            print(log_entry.strip())

print(f"\nsimulation finished. Check {LOG_FILE} for full history.")