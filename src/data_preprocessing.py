import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

def preprocess_hybrid_lupus(clinical_path, genomic_path):
    """
    Combines clinical and genomic data for early lupus prediction in-memory.
    """
    print("Step 1: loading clinical csv...", flush=True)
    df_clin = pd.read_csv(clinical_path)

    print("Step 2: creating target and clinical features...", flush=True)
    if 'Classification Score' in df_clin.columns:
        y = (df_clin['Classification Score'] >= 10).astype(int).values
        X_clin = df_clin.drop(['Classification Score', 'PROV', 'Gender'], axis=1, errors='ignore')
    else:
        raise ValueError("Target column 'Classification Score' not found!")

    print("Step 3: loading genomic txt...", flush=True)
    df_gen = pd.read_csv(genomic_path, sep='\t', comment='!')

    print("Step 4: setting index and transpose...", flush=True)
    df_gen = df_gen.set_index('ID_REF').T

    potential_probes = [
        'ILMN_1343291', 'ILMN_1343295', 'ILMN_1651210',
        'ILMN_1707010', 'ILMN_1721245', 'ILMN_1804139',
        'ILMN_1651209', 'ILMN_1651199'
    ]

    print("Step 5: selecting genomic probes...", flush=True)
    available_probes = [p for p in potential_probes if p in df_gen.columns]

    if not available_probes:
        raise ValueError("None of the required Lupus genes were found in the genomic file!")

    print(f"Found {len(available_probes)} valid genomic markers.", flush=True)
    X_gen_selected = df_gen[available_probes]

    print("Step 6: matching and combining...", flush=True)
    X_gen_matched = X_gen_selected.iloc[:len(X_clin)].reset_index(drop=True)
    X_clin_clean = X_clin.reset_index(drop=True).apply(pd.to_numeric, errors='coerce').fillna(0)
    X_combined = pd.concat([X_clin_clean, X_gen_matched], axis=1)

    print("Step 7: scaling...", flush=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    print("Step 8: PCA...", flush=True)
    pca = PCA(n_components=6)
    X_reduced = pca.fit_transform(X_scaled)

    print("Step 9: MinMax scaling...", flush=True)
    q_scaler = MinMaxScaler(feature_range=(0, 1))
    X_final = q_scaler.fit_transform(X_reduced)

    print("Step 10: train-test split...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y
    )

    total_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"📊 Information Retained: {total_variance:.2f}%", flush=True)

    if np.isnan(X_final).any():
        print("❌ Warning: Missing values detected!", flush=True)
    else:
        print("✅ Integrity Check: Clean and complete.", flush=True)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    CLINICAL_FILE = os.path.abspath(os.path.join(base_path, "..", "dataset", "lupus_dataset.csv"))
    GENOMIC_FILE = os.path.abspath(os.path.join(base_path, "..", "dataset", "GSE65391_medium.txt"))

    try:
        X_train, X_test, y_train, y_test = preprocess_hybrid_lupus(CLINICAL_FILE, GENOMIC_FILE)
        print(f"🚀 Data Loaded for Model Training. Samples: {X_train.shape[0]}", flush=True)
    except Exception as e:
        print(f"❌ An error occurred: {e}", flush=True)