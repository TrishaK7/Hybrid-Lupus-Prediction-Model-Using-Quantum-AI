import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def train_with_abstention(X, y):
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = SVC(kernel="rbf", probability=True, random_state=42)
    model.fit(X_train, y_train)

    cal_probs = model.predict_proba(X_cal)
    nonconformity = 1 - cal_probs[np.arange(len(y_cal)), y_cal]
    qhat = np.quantile(nonconformity, 0.90)

    print(f"📊 Abstention threshold (qhat): {qhat:.4f}", flush=True)
    return model, qhat


def predict_with_abstention(model, qhat, X_new):
    probs = model.predict_proba(X_new)
    preds = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)

    outputs = []
    for i in range(len(X_new)):
        score = 1 - probs[i, preds[i]]
        confidence = round(float(conf[i]), 4)

        if score > qhat:
            outputs.append(("UNCERTAIN", confidence))
        else:
            outputs.append((int(preds[i]), confidence))

    return outputs