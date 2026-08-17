from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
TARGET_COLUMN = "target"
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # sklearn uses 0=malignant, 1=benign.
    df[TARGET_COLUMN] = data.target
    return df


def get_models():
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        # The assignment text says "6 ML models" but lists 5. SVM is added as a safe sixth model.
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(C=2.0, kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            ]
        ),
    }


def model_file_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_") + ".joblib"


def predict_scores(model, x_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x_test)
    return model.predict(x_test)


def evaluate_model(model, x_test, y_test) -> dict:
    y_pred = model.predict(x_test)
    y_score = predict_scores(model, x_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    df.to_csv(ROOT / "breast_cancer_uci_full.csv", index=False)

    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    test_df = x_test.copy()
    test_df[TARGET_COLUMN] = y_test.values
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    rows = []
    for model_name, model in get_models().items():
        model.fit(x_train, y_train)
        joblib.dump(model, MODEL_DIR / model_file_name(model_name))
        metrics = evaluate_model(model, x_test, y_test)
        rows.append({"ML Model Name": model_name, **metrics})

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(ROOT / "metrics_summary.csv", index=False)

    metadata = {
        "dataset_name": "UCI Breast Cancer Wisconsin Diagnostic",
        "source": "sklearn.datasets.load_breast_cancer, originally from UCI Machine Learning Repository",
        "target_column": TARGET_COLUMN,
        "target_mapping": {"0": "malignant", "1": "benign"},
        "total_instances": int(df.shape[0]),
        "total_features": int(x.shape[1]),
        "test_instances": int(test_df.shape[0]),
        "models": list(get_models().keys()),
        "random_state": RANDOM_STATE,
    }
    (ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Generated files:")
    print(f"- {ROOT / 'breast_cancer_uci_full.csv'}")
    print(f"- {ROOT / 'test_data.csv'}")
    print(f"- {ROOT / 'metrics_summary.csv'}")
    print(f"- {ROOT / 'metadata.json'}")
    print(f"- {MODEL_DIR}/*.joblib")
    print()
    print(metrics_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
