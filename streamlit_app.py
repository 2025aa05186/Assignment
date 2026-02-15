import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="ML Classifier", layout="wide")
st.title("Machine Learning Classification App")


# ----------------------------
# Load bundle
# ----------------------------
@st.cache_resource
def load_bundle():
    return joblib.load("models.pkl")


bundle = load_bundle()

models = {
    "Logistic Regression": bundle["logistic_regression"],
    "Decision Tree": bundle["decision_tree"],
    "KNN": bundle["knn"],
    "Naive Bayes": bundle["naive_bayes"],
    "Random Forest": bundle["random_forest"],
    "XGBoost": bundle["xgboost"],
}

scaler = bundle["scaler"]
imputer = bundle["imputer"]
encoders = bundle["encoders"]
target_encoder = bundle["target_encoder"]
feature_columns = bundle["feature_columns"]


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Controls")

model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
model = models[model_name]

uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])


# ----------------------------
# Prediction block
# ----------------------------
if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(data.head())

    # last column is target
    X = data.iloc[:, :-1].copy()
    y = data.iloc[:, -1].copy()

    # reorder columns
    X = X[feature_columns]

    # encode categorical
    for col in encoders:
        X[col] = encoders[col].transform(X[col].astype(str))

    # impute
    X = imputer.transform(X)

    # scale
    X = scaler.transform(X)

    # encode target
    y = target_encoder.transform(y)

    # prediction
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
            auc = "N/A"
    except:
        auc = "N/A"

    st.subheader("Evaluation Metrics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc:.4f}")
    c1.metric("Precision", f"{prec:.4f}")
    c2.metric("Recall", f"{rec:.4f}")
    c2.metric("F1 Score", f"{f1:.4f}")
    c3.metric("MCC", f"{mcc:.4f}")
    c3.metric("AUC", auc if isinstance(auc, str) else f"{auc:.4f}")

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
    st.info("Upload a CSV file from sidebar.")


# ----------------------------
# Footer
# ----------------------------
st.write("---")
st.write("Developed for ML Assignment")
