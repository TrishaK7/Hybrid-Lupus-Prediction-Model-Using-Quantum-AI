import pennylane as qml
from pennylane import numpy as pnp
import numpy as np

# Number of qubits (matches PCA components)
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# ----------------------------
# Quantum Circuit
# ----------------------------
def quantum_circuit(inputs, weights):
    # Encode classical features into qubit rotations
    for i in range(n_qubits):
        qml.RY(np.pi * inputs[i], wires=i)
    # Variational entangling layers
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    # Measure first qubit for binary classification
    return qml.expval(qml.PauliZ(0))

qnode = qml.QNode(quantum_circuit, dev)

# ----------------------------
# Cost function
# ----------------------------
def cost(weights, X, y):
    # Map expectation [-1,1] to probability [0,1]
    preds = [0.5 * (1 - qnode(x, weights)) for x in X]
    preds = np.clip(preds, 1e-6, 1-1e-6)  # numerical stability
    # Binary cross-entropy
    return -np.mean(y*np.log(preds) + (1-y)*np.log(1-preds))

# ----------------------------
# Prediction
# ----------------------------
def predict(weights, X):
    probs = [0.5 * (1 - qnode(x, weights)) for x in X]
    return np.array(probs) > 0.5

# ----------------------------
# Training Function
# ----------------------------
def train_quantum_model(X_train, y_train, layers=3, epochs=50, lr=0.1):
    """
    Trains a Variational Quantum Classifier.
    Returns the trained weights.
    """
    weights = pnp.random.randn(layers, n_qubits, 3, requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=lr)

    print("🚀 Training Quantum Model...")
    for epoch in range(epochs):
        weights, loss = opt.step_and_cost(lambda w: cost(w, X_train, y_train), weights)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return weights