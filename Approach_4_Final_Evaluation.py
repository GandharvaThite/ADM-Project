import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

warnings.filterwarnings('ignore')

print("=" * 100)
print("       APPROACH 4: FINAL BEHAVIORAL CASCADE EVALUATION (2017 & 2018)")
print("=" * 100)

# 1. Define Paths & Load Final Artifact
ART_PATH = r"X:\M.E\Sem1\ADM\ADM\preprocessing\Tier2_finalFailure\Final_ME_Architecture.pkl"
PATH_2017 = r"X:\M.E\Sem1\ADM\ADM\preprocessing\CIC-IDS2018-CSVs\cicids2017_curated_38cols_Binary_Latest.csv"
PATH_2018 = r"X:\M.E\Sem1\ADM\ADM\preprocessing\CIC-IDS2018-CSVs\cicids2018_curated_balanced.csv"

print("[*] Loading Final Pipeline Artifact...")
art = joblib.load(ART_PATH)
scaler = art["scaler"]
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

# 2. Load 2017 Dataset
print("[*] Loading 2017 In-Distribution Dataset...")
df_17 = pd.read_csv(PATH_2017, low_memory=False)
df_17.columns = df_17.columns.str.strip()
lbl_col_17 = 'Label' if 'Label' in df_17.columns else df_17.columns[-1]

y_17 = df_17[lbl_col_17].to_numpy(dtype=np.int32)
for c in orig_feature_cols:
    df_17[c] = pd.to_numeric(df_17[c], errors='coerce')
df_17.replace([np.inf, -np.inf], np.nan, inplace=True)
df_17.dropna(subset=orig_feature_cols, inplace=True)
y_17 = y_17[df_17.index]
X_17_raw = np.nan_to_num(df_17[orig_feature_cols].to_numpy(dtype=np.float64), nan=0.0, posinf=1e9, neginf=-1e9)

# 3. Ingest 2018 Dataset (With Translation)
print("[*] Loading 2018 Unseen Zero-Day Dataset...")
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

df_18 = pd.read_csv(PATH_2018, low_memory=False)
df_18.columns = df_18.columns.str.strip()
df_18 = df_18[df_18['Label'].astype(str).str.strip().str.upper() != 'LABEL'].copy()

for col in list(set(cols_18)):
    df_18[col] = pd.to_numeric(df_18[col], errors='coerce')
df_18.replace([np.inf, -np.inf], np.nan, inplace=True)
df_18.dropna(subset=list(set(cols_18)) + ['Label'], inplace=True)

raw_lbl_18 = df_18['Label'].astype(str).str.strip()
y_18 = (raw_lbl_18.str.upper() != 'BENIGN').astype(int).values
X_18_raw = np.nan_to_num(df_18[cols_18].to_numpy(dtype=np.float64), nan=0.0, posinf=1e9, neginf=-1e9)

datasets = {
    "2017 (In-Distribution)": (X_17_raw, y_17),
    "2018 (Unseen Zero-Day)": (X_18_raw, y_18)
}

# 4. Evaluation Function with Full Class Separation
def evaluate_cascade(X_raw, y_true):
    start_time = time.perf_counter()
    
    # 1. Scale
    X_scaled = scaler.transform(X_raw)
    
    # 2. Geometric Feature Extraction
    dists = cdist(X_scaled, centroids, metric='euclidean')
    nr = np.argmin(dists, axis=1)
    eff_s = np.maximum(cluster_stds[nr], np.maximum(0.10 * global_std, 0.05))
    
    z_mat = np.abs(X_scaled - centroids[nr]) / eff_s
    min_dist = np.min(dists, axis=1, keepdims=True)
    top3_z = np.mean(np.partition(z_mat, -3, axis=1)[:, -3:], axis=1, keepdims=True)
    max_z = np.max(z_mat, axis=1, keepdims=True)
    comp_scores = (0.60 * (top3_z / m_top3) + 0.40 * (max_z / m_max)).ravel()
    
    # 3. Tier 1 Fast Drop
    t1_drop = (comp_scores >= t1_cutoff)
    ambig_mask = ~t1_drop
    
    # 4. Tier 2 Behavioral Arbiter
    t2_catch = np.zeros(len(y_true), dtype=bool)
    if np.any(ambig_mask):
        X_behavioral = X_scaled[ambig_mask][:, keep_indices]
        X_aug = np.hstack([X_behavioral, min_dist[ambig_mask], top3_z[ambig_mask], max_z[ambig_mask], comp_scores[ambig_mask].reshape(-1, 1)])
        gb_probs = gb_arbiter.predict_proba(X_aug)[:, 1]
        t2_catch[ambig_mask] = (gb_probs >= t2_threshold)
        
    y_pred = (t1_drop | t2_catch).astype(int)
    
    # Latency Calculation
    total_time_sec = time.perf_counter() - start_time
    latency_us = (total_time_sec / len(y_true)) * 1e6
    
    # Metrics Separation
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    overall_acc = (tp + tn) / len(y_true) * 100
    fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    
    # Malign Metrics (Class 1)
    rec_malign = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    prec_malign = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    f1_malign = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # Benign Metrics (Class 0)
    rec_benign = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    prec_benign = tn / (tn + fn) * 100 if (tn + fn) > 0 else 0
    f1_benign = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    return overall_acc, fpr, latency_us, rec_malign, prec_malign, f1_malign, rec_benign, prec_benign, f1_benign

# 5. Run & Display Results
print("\n" + "=" * 100)
print(f"{'Dataset Phase':<25} | {'Class':<8} | {'Precision (%)':<15} | {'Recall/Acc (%)':<15} | {'F1-Score':<10}")
print("-" * 100)

for ds_name, (X, y) in datasets.items():
    (overall_acc, fpr, lat, rec_mal, prec_mal, f1_mal, rec_ben, prec_ben, f1_ben) = evaluate_cascade(X, y)
    
    print(f"{ds_name:<25} | {'Benign':<8} | {prec_ben:<15.2f} | {rec_ben:<15.2f} | {f1_ben:<10.4f}")
    print(f"{'':<25} | {'Malign':<8} | {prec_mal:<15.2f} | {rec_mal:<15.2f} | {f1_mal:<10.4f}")
    print(f"    -> Overall Acc: {overall_acc:.2f}% | FPR: {fpr:.2f}% | Latency: {lat:.2f} us/flow\n")
    
print("=" * 100)