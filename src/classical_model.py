from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

from data_preprocessing import preprocess_lupus_data

def train_classical_model():


    # Load preprocessed data
    X_train, X_test, y_train, y_test = preprocess_lupus_data('dataset/lupus_dataset.csv')

    # Train SVM model
    model = SVC(kernel='rbf', probability=True, random_state=42)

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Cross Validation
    scores = cross_val_score(model, X_train, y_train, cv=5)

    print("\nCross Validation Scores:", scores)
    print("Average CV Accuracy:", scores.mean())

    print("\n🎯 Classical SVM Model Results")
    print("----------------------------")
    print("Accuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.show()

    return model


if __name__ == "__main__":
    train_classical_model()  