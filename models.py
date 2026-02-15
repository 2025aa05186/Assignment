import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ----------------------------
# Load dataset
# ----------------------------
import kagglehub

# Download latest version
path = kagglehub.dataset_download("rajyellow46/wine-quality")

print("Path to dataset files:", path)
df = pd.read_csv("train.csv")   # change if needed

# last column = target
X = df.iloc[:, :-1].copy()
y = df.iloc[:, -1].copy()

feature_columns = list(X.columns)


# ----------------------------
# Encode categorical features
# ----------------------------
encoders = {}

for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le


# ----------------------------
# Handle missing values
# ----------------------------
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)


# ----------------------------
# Scale
# ----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)


# ----------------------------
# Encode target (IMPORTANT for XGBoost)
# ----------------------------
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)


# ----------------------------
# Train Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ----------------------------
# Initialize Models
# ----------------------------
lr = LogisticRegression(max_iter=500)
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()
nb = GaussianNB()
rf = RandomForestClassifier()
xgb = XGBClassifier(eval_metric="logloss")


# ----------------------------
# Train
# ----------------------------
lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
knn.fit(X_train, y_train)
nb.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)


# ----------------------------
# Bundle Everything
# ----------------------------
bundle = {
    "logistic_regression": lr,
    "decision_tree": dt,
    "knn": knn,
    "naive_bayes": nb,
    "random_forest": rf,
    "xgboost": xgb,
    "scaler": scaler,
    "imputer": imputer,
    "encoders": encoders,
    "target_encoder": target_encoder,
    "feature_columns": feature_columns,
}


# ----------------------------
# Save ONE file
# ----------------------------
joblib.dump(bundle, "models.pkl")

print("✅ models.pkl created successfully!")