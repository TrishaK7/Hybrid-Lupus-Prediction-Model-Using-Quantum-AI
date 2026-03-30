print("TOP OF FILE REACHED", flush=True)

import os
print("IMPORTING PREPROCESSING...", flush=True)
from data_preprocessing import preprocess_hybrid_lupus

print("IMPORTING UNCERTAINTY MODULE...", flush=True)
from uncertainty_model import train_with_abstention, predict_with_abstention

print("ALL IMPORTS DONE", flush=True)

base_path = os.path.dirname(os.path.abspath(__file__))
CLINICAL_FILE = os.path.abspath(os.path.join(base_path, "..", "dataset", "lupus_dataset.csv"))
GENOMIC_FILE = os.path.abspath(os.path.join(base_path, "..", "dataset", "GSE65391_series_matrix.txt"))

print("CLINICAL:", CLINICAL_FILE, flush=True)
print("GENOMIC:", GENOMIC_FILE, flush=True)

print("CALLING PREPROCESS...", flush=True)
X_train, X_test, y_train, y_test = preprocess_hybrid_lupus(CLINICAL_FILE, GENOMIC_FILE)
print("PREPROCESS DONE", flush=True)

print("TRAINING UNCERTAINTY MODEL...", flush=True)
model, qhat = train_with_abstention(X_train, y_train)
print("UNCERTAINTY MODEL DONE", flush=True)

print("PREDICTING...", flush=True)
results = predict_with_abstention(model, qhat, X_test[:10])
print("PREDICTION DONE", flush=True)

print("\n🔍 Predictions with Uncertainty:")
for i, result in enumerate(results, 1):
    pred, conf = result

    if pred == "UNCERTAIN":
        label = "Needs Review"
    elif pred == 1:
        label = "Lupus Risk"
    else:
        label = "Low Risk"

    print(f"Sample {i}: {label} | Confidence: {conf * 100:.2f}%")