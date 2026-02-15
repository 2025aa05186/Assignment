import streamlit as st
import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)

import seaborn as sns
import matplotlib.pyplot as plt


# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="ML Classification App", layout="wide")

st.title("Machine Learning Classification & Evaluation")
st.write("Upload test dataset → Select model → View performance")


# ----------------------------
# Load Models
# ----------------------------
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
        "Decision Tree": joblib.load("models/decision_tree.pkl"),
        "KNN": joblib.load("models/knn.pkl"),
        "Naive Bayes": joblib.load("models/naive_bayes.pkl"),
        "Random Forest": joblib.load("models/random_forest.pkl"),
        "XGBoost": joblib.load("models/xgboost.pkl"),
    }
    return models


models = load_models()


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Controls")

model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
model = models[model_name]


uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])


# ----------------------------
# Main Logic
# ----------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(data.head())

    # assuming last column is target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Prediction
    y_pred = model.predict(X)

    # ----------------------------
    # Metrics
    # ----------------------------
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y, y_pred)

    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)
            auc = roc_auc_score(y, y_prob, multi_class="ovr")
        else:
            auc = "Not Available"
    except:
        auc = "Not Available"

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{acc:.4f}")
    col1.metric("Precision", f"{prec:.4f}")

    col2.metric("Recall", f"{rec:.4f}")
    col2.metric("F1 Score", f"{f1:.4f}")

    col3.metric("MCC", f"{mcc:.4f}")
    col3.metric("AUC", auc if isinstance(auc, str) else f"{auc:.4f}")

    # ----------------------------
    # Confusion Matrix
    # ----------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)


else:
    st.info("Please upload a CSV file to begin.")


# ----------------------------
# Footer
# ----------------------------
st.write("---")
st.write("Developed for ML Assignment Deployment")