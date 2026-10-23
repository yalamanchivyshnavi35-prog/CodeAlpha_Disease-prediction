
# Disease Prediction Using Machine Learning
# Algorithms: Logistic Regression, SVM, Random Forest, XGBoost

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------
# 1. LOAD MEDICAL DATASET
# ------------------------------------------------
data = load_breast_cancer()

X = data.data
y = data.target

# Dataset target: 0 = malignant, 1 = benign
# This dataset contains measurements of cell nuclei.
print("Dataset loaded successfully")
print("Number of patient records:", X.shape[0])
print("Number of features:", X.shape[1])
print("Target classes:", list(data.target_names))

# ------------------------------------------------
# 2. SPLIT DATA INTO TRAINING AND TESTING
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ------------------------------------------------
# 3. DEFINE CLASSIFICATION MODELS
# ------------------------------------------------
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC())
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

# Add XGBoost if installed
try:
    from xgboost import XGBClassifier

    models["XGBoost"] = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42
    )
except ImportError:
    print("XGBoost is not installed; skipping XGBoost.")

# ------------------------------------------------
# 4. TRAIN AND EVALUATE MODELS
# ------------------------------------------------
results = {}

for name, model in models.items():
    print("\n" + "=" * 50)
    print("Training:", name)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy

    print("Test accuracy:", round(accuracy * 100, 2), "%")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=data.target_names,
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

# ------------------------------------------------
# 5. COMPARE MODEL RESULTS
# ------------------------------------------------
print("\nMODEL COMPARISON")
print("-" * 40)

for name, accuracy in sorted(
    results.items(),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{name}: {accuracy * 100:.2f}%")

# ------------------------------------------------
# 6. PREDICT A SAMPLE FROM THE TEST SET
# ------------------------------------------------
# This demonstrates the prediction process using
# a held-out example from the dataset.
selected_model_name = max(results, key=results.get)
best_model = models[selected_model_name]

sample = X_test[0].reshape(1, -1)
prediction = int(best_model.predict(sample)[0])

print("\nSAMPLE PREDICTION DEMONSTRATION")
print("Model:", selected_model_name)
print("Predicted dataset category:", data.target_names[prediction])
print("Actual dataset category:", data.target_names[y_test[0]])

print("\nProject completed successfully.")
