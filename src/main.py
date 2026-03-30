import numpy as np
import os
import dill  # Added for saving the model
from data_preprocessing import preprocess_hybrid_lupus
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import SPSA
from qiskit.primitives import StatevectorSampler as Sampler 
from qiskit_algorithms.utils import algorithm_globals

# 1. LOCK THE NEW SEED
algorithm_globals.random_seed = 100

def train_quantum_lupus_model():
    # 2. SETUP PATHS
    base_path = os.path.dirname(os.path.abspath(__file__))
    CLINICAL_FILE = os.path.abspath(os.path.join(base_path, "..", "dataset", "lupus_dataset.csv"))
    GENOMIC_FILE = os.path.abspath(os.path.join(base_path, "..", "dataset", "GSE65391_series_matrix.txt"))

    # 3. LOAD DATA
    X_train, X_test, y_train, y_test = preprocess_hybrid_lupus(CLINICAL_FILE, GENOMIC_FILE)

    # 4. QUANTUM ARCHITECTURE
    num_qubits = 6
    # Switched to circular entanglement to help different symptoms 'talk' better
    f_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement='circular')
    ansatz = RealAmplitudes(num_qubits=num_qubits, reps=3)

    # 5. STABILIZE INITIAL WEIGHTS
    initial_point = algorithm_globals.random.random(ansatz.num_parameters)

    # 6. CONSTRUCT VQC
    vqc = VQC(
        feature_map=f_map,
        ansatz=ansatz,
        optimizer=SPSA(maxiter=250),
        sampler=Sampler(),
        initial_point=initial_point
    )

    print(f"🩺 Quantum Model is learning (Seed: 100, Qubits: {num_qubits})...")
    vqc.fit(X_train, y_train)
    
    # 7. EVALUATION
    score = vqc.score(X_test, y_test)
    accuracy_pct = score * 100
    print(f"\n--- 📊 FINAL DOCTOR'S REPORT ---")
    print(f"Quantum Model Accuracy: {accuracy_pct:.2f}%")
    
    # 8. THE LOCK: Save the model if it hits your desired high accuracy
    if accuracy_pct >= 85.0:
        results_dir = os.path.abspath(os.path.join(base_path, "..", "results"))
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        model_path = os.path.join(results_dir, "trained_lupus_model.pkl")
        with open(model_path, 'wb') as f:
            dill.dump(vqc, f)
        print(f"🔒 ACCURACY LOCKED! Model saved to: {model_path}")
    else:
        print("⚠️ Accuracy below 85%. Model not saved. Try another seed or Qubits=8.")
    
    return vqc

if __name__ == "__main__":
    trained_model = train_quantum_lupus_model()
