from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET_COLUMN = "target"


st.set_page_config(page_title="Breast Cancer Classification Lab", layout="wide")
st.title("Breast Cancer Classification Lab")
st.caption("BITS ML Assignment 2 - model comparison and deployment demo")


@st.cache_data
def load_metadata():
    metadata_path = ROOT / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "dataset_name": "UCI Breast Cancer Wisconsin Diagnostic",
        "target_column": TARGET_COLUMN,
        "target_mapping": {"0": "malignant", "1": "benign"},
    }


@st.cache_resource
def load_models():
    models = {}
    for path in sorted(MODEL_DIR.glob("*.joblib")):
        model_name = path.stem.replace("_", " ").title()
        models[model_name] = joblib.load(path)
    return models


def model_score(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return model.predict(x)


def calculate_metrics(model, data):
    x = data.drop(columns=[TARGET_COLUMN])
    y_true = data[TARGET_COLUMN]
    y_pred = model.predict(x)
    y_score = model_score(model, x)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }, y_true, y_pred


metadata = load_metadata()
models = load_models()

with st.sidebar:
    st.header("Dataset")
    st.write(metadata.get("dataset_name", "Classification dataset"))
    st.write(f"Target column: `{metadata.get('target_column', TARGET_COLUMN)}`")
    st.write("Target mapping: `0 = malignant`, `1 = benign`")
    st.divider()
    st.write("Upload `test_data.csv` or use the bundled test split.")


if not models:
    st.error("No trained models found. Run `python model/train_models.py` before deploying.")
    st.stop()


uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])
if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
else:
    default_test_path = ROOT / "test_data.csv"
    if not default_test_path.exists():
        st.error("No `test_data.csv` found. Run `python model/train_models.py` first.")
        st.stop()
    test_data = pd.read_csv(default_test_path)


if TARGET_COLUMN not in test_data.columns:
    st.error(f"The uploaded CSV must contain the target column `{TARGET_COLUMN}`.")
    st.stop()


model_name = st.selectbox("Select model", list(models.keys()))
selected_model = models[model_name]

st.subheader("Uploaded Test Data Preview")
st.dataframe(test_data.head(20), use_container_width=True)

metrics, y_true, y_pred = calculate_metrics(selected_model, test_data)

st.subheader(f"Evaluation Metrics - {model_name}")
metric_cols = st.columns(6)
for col, (metric_name, value) in zip(metric_cols, metrics.items()):
    col.metric(metric_name, f"{value:.4f}")

left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)


st.subheader("All Model Comparison")
comparison_rows = []
for name, model in models.items():
    model_metrics, _, _ = calculate_metrics(model, test_data)
    comparison_rows.append({"ML Model Name": name, **model_metrics})

comparison_df = pd.DataFrame(comparison_rows)
st.dataframe(comparison_df.style.format({c: "{:.4f}" for c in comparison_df.columns if c != "ML Model Name"}), use_container_width=True)

best_row = comparison_df.sort_values(["F1", "MCC", "AUC"], ascending=False).iloc[0]
st.success(
    f"Overall winner on this test split: {best_row['ML Model Name']} "
    f"(F1={best_row['F1']:.4f}, MCC={best_row['MCC']:.4f}, AUC={best_row['AUC']:.4f})."
)
