import numpy as np

def dynamic_ensemble_predict(vqc_prediction, classical_probs):
    """
    Combines:
    - quantum VQC class prediction
    - classical model probabilities

    Returns:
    - final_pred
    - hybrid_lupus_prob
    - hybrid_low_risk_prob
    """

    # Classical probabilities
    low_prob = float(classical_probs[0])
    high_prob = float(classical_probs[1])

    # Convert quantum class prediction into soft probability
    # Minimal-change approach:
    # if VQC predicts lupus => give strong support to class 1
    # if VQC predicts low risk => give strong support to class 0
    if int(vqc_prediction) == 1:
        quantum_probs = np.array([0.20, 0.80])
    else:
        quantum_probs = np.array([0.80, 0.20])

    classical_prob_array = np.array([low_prob, high_prob])

    # Confidence-based dynamic weighting
    classical_conf = np.max(classical_prob_array)
    quantum_conf = np.max(quantum_probs)

    total_conf = classical_conf + quantum_conf + 1e-8
    classical_weight = classical_conf / total_conf
    quantum_weight = quantum_conf / total_conf

    hybrid_probs = (
        classical_weight * classical_prob_array +
        quantum_weight * quantum_probs
    )

    final_pred = int(np.argmax(hybrid_probs))
    hybrid_low_risk_prob = float(hybrid_probs[0]) * 100
    hybrid_lupus_prob = float(hybrid_probs[1]) * 100

    return final_pred, hybrid_lupus_prob, hybrid_low_risk_prob, classical_weight, quantum_weight