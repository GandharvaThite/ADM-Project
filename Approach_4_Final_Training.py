import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings('ignore')

print("=" * 85)
print("    SCRIPT 1: TRAIN & EXPORT FINAL APPROACH 4 (TWO-TIER )")
print("=" * 85)

# 1. Path Configuration
CURRENT_DIR = os.getcwd()
MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, "Final_ME_Architecture.pkl")

CANDIDATE_DATA_PATHS = [
    os.path.join(CURRENT_DIR, "cicids2017_curated_38cols_Binary_Latest.csv"),
    r"O:\ADM\cicids2017_curated_38cols_Binary_Latest.csv",
    r"X:\M.E\Sem1\ADM\ADM\preprocessing\CIC-IDS2018-CSVs\cicids2017_curated_38cols_Binary_Latest.csv"
]

DATA_PATH = None
for path in CANDIDATE_DATA_PATHS:
    if os.path.exists(path):
        DATA_PATH = path
        break

if DATA_PATH is None:
    raise FileNotFoundError("Could not find 'cicids2017_curated_38cols_Binary_Latest.csv'.")

# 2. Ingest 2017 Training Dataset
print(f"[*] Ingesting 2017 curated dataset: {DATA_PATH}...")
df_17 = pd.read_csv(DATA_PATH, low_memory=False)
df_17.columns = df_17.columns.str.strip()

lbl_col = 'Label' if 'Label' in df_17.columns else df_17.columns[-1]

raw_lbl = df_17[lbl_col]
if pd.api.types.is_numeric_dtype(raw_lbl):
    y_binary = (raw_lbl.values != 0).astype(int)
else:
    s_lbl = raw_lbl.astype(str).str.strip().str.upper()
    y_binary = (~s_lbl.isin(['BENIGN', '0', '0.0', 'NORMAL'])).astype(int).values

feature_cols = [c for c in df_17.columns if c != lbl_col and np.issubdtype(df_17[c].dtype, np.number)]
print(f"[+] Loaded {len(df_17):,} flows with {len(feature_cols)} numeric features.")

X_raw = np.nan_to_num(df_17[feature_cols].to_numpy(dtype=np.float64), nan=0.0, posinf=1e9, neginf=-1e9)

# STARTING PIPELINE TRAINING TIMER
pipeline_start_time = time.perf_counter()

# 3. Robust Scaling on Benign Baseline
print("[*] Fitting RobustScaler on benign baseline distribution...")
scaler = RobustScaler()
scaler.fit(X_raw[y_binary == 0])
X_scaled = scaler.transform(X_raw)

# 4. Tier 1: LSH Partitioning & Intra-Bucket DBSCAN
print("[*] Tier 1: LSH Partitioning via Gaussian Random Projections...")
benign_train_pts = X_scaled[y_binary == 0]

if len(benign_train_pts) > 30000:
    rng = np.random.default_rng(42)
    benign_train_pts = benign_train_pts[rng.choice(len(benign_train_pts), 30000, replace=False)]

n_bits = 5 
grp = GaussianRandomProjection(n_components=n_bits, random_state=42)
projections = grp.fit_transform(benign_train_pts)
bucket_ids = np.dot((projections > 0).astype(int), 1 << np.arange(n_bits)[::-1])

print("Running Intra-Bucket DBSCAN (eps=3.0, min_samples=5)...")
lsh_centroids, lsh_stds = [], []

for b_id in range(2**n_bits):
    b_pts = benign_train_pts[bucket_ids == b_id]
    if len(b_pts) < 10: continue
    
    db = DBSCAN(eps=3.0, min_samples=5, metric='euclidean', n_jobs=-1)
    labels = db.fit_predict(b_pts)
    
    core = b_pts[labels >= 0]
    if len(core) >= 5:
        lsh_centroids.append(np.mean(core, axis=0))
        lsh_stds.append(np.maximum(np.std(core, axis=0), 1e-3))
    else:
        lsh_centroids.append(np.mean(b_pts, axis=0))
        lsh_stds.append(np.maximum(np.std(b_pts, axis=0), 1e-3))

centroids = np.array(lsh_centroids)
cluster_stds = np.array(lsh_stds)
global_std = np.std(benign_train_pts, axis=0)

print("Calibrating Tier-1 normalizers...")
dists = cdist(benign_train_pts, centroids, metric='euclidean')
nearest = np.argmin(dists, axis=1)

eff_std = np.maximum(cluster_stds[nearest], np.maximum(0.10 * global_std, 0.05))
z_mat = np.minimum(np.abs(benign_train_pts - centroids[nearest]) / eff_std, 25.0)

top3_z = np.mean(np.partition(z_mat, -3, axis=1)[:, -3:], axis=1)
max_z = np.max(z_mat, axis=1)

med_top3 = float(np.median(top3_z))
med_max = float(np.median(max_z))

T1_CUTOFF = 196.65  
print(f"Enforcing strict Tier 1 Geometric Cutoff: {T1_CUTOFF}")

# 5. Feature Pruning (Dropping 9 Features)
# =========================================================================
print("Pruning 9 network-specific/collinear features for Generalization...")
variances = np.var(X_scaled, axis=0)
drop_indices = np.argsort(variances)[:9]
keep_indices = np.setdiff1d(np.arange(len(feature_cols)), drop_indices)

print(f"Compressed behavioral vector to {len(keep_indices)} dimensions.")

# 6. Prepare 32-D Vector & Train Tier 2 Arbiter
# =========================================================================
print("Generating Augmented 32-D Embeddings for Tier 2 Arbiter...")
dists_all = cdist(X_scaled, centroids, metric='euclidean')
nr_all = np.argmin(dists_all, axis=1)
eff_s_all = np.maximum(cluster_stds[nr_all], np.maximum(0.10 * global_std, 0.05))

z_mat_all = np.abs(X_scaled - centroids[nr_all]) / eff_s_all
min_dist_all = np.min(dists_all, axis=1, keepdims=True)
top3_z_all = np.mean(np.partition(z_mat_all, -3, axis=1)[:, -3:], axis=1, keepdims=True)
max_z_all = np.max(z_mat_all, axis=1, keepdims=True)
comp_scores = (0.60 * (top3_z_all / med_top3) + 0.40 * (max_z_all / med_max))

X_beh = X_scaled[:, keep_indices]
X_aug = np.hstack([X_beh, min_dist_all, top3_z_all, max_z_all, comp_scores])

print("Training Gradient Boosting Arbiter (N=200 Estimators)...")
benign_idx = np.where(y_binary == 0)[0]
malicious_idx = np.where(y_binary == 1)[0]
np.random.seed(42)
sample_b = np.random.choice(benign_idx, size=min(125000, len(benign_idx)), replace=False)
sample_m = np.random.choice(malicious_idx, size=min(125000, len(malicious_idx)), replace=False)
train_idx = np.concatenate([sample_b, sample_m])

gb_arbiter = HistGradientBoostingClassifier(
    max_iter=200, 
    learning_rate=0.1, 
    max_depth=12,
    min_samples_leaf=20,
    l2_regularization=0.5,
    random_state=42
)
gb_arbiter.fit(X_aug[train_idx], y_binary[train_idx])

T2_THRESHOLD = 0.20
print(f"Gradient Boosting trained. Setting Zero-Day Threshold to {T2_THRESHOLD}")

# STOPPING PIPELINE TRAINING TIMER
# =========================================================================
pipeline_end_time = time.perf_counter()
training_duration = pipeline_end_time - pipeline_start_time
mins, secs = divmod(training_duration, 60)

# 7. Saving Final Architecture
# =========================================================================
pipeline_bundle = {
    "scaler": scaler,
    "centroids": centroids,
    "cluster_stds": cluster_stds,
    "global_std": global_std,
    "m_top3": med_top3,
    "m_max": med_max,
    "orig_feature_cols": feature_cols,
    "keep_indices": keep_indices,
    "gb_arbiter": gb_arbiter,
    "t1_cutoff": T1_CUTOFF,
    "t2_threshold": T2_THRESHOLD
}

joblib.dump(pipeline_bundle, MODEL_SAVE_PATH)
print("\n" + "=" * 85)
print(f"TOTAL ARCHITECTURE TRAINING TIME: {int(mins)} minutes and {secs:.2f} seconds")
print(f"FINAL PIPELINE EXPORTED SUCCESSFULLY:\n    --> {MODEL_SAVE_PATH}")
print("=" * 85)