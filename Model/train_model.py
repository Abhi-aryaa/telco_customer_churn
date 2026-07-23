import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from pathlib import Path

SEED = 42
THRESHOLD = 0.45  # chosen by precision/recall/F1 tradeoff

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR.parent / "Dataset" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_OUT_PATH = SCRIPT_DIR / "churn_model.joblib"

# ---------------------------------------------------------------
# 1. Load & clean data 
# ---------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)
df = df.drop(columns=["customerID"])

df["Churn"] = (df["Churn"] == "Yes").astype(int)

addon_cols = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
for col in addon_cols:
    df[col] = df[col].replace("No internet service", "No")

df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

# ---------------------------------------------------------------
# 2. Feature engineering 
# ---------------------------------------------------------------
df["num_addons"] = (df[addon_cols] == "Yes").sum(axis=1)
df["charge_per_addon"] = (df["MonthlyCharges"] / (df["num_addons"] + 1)).round(2)

TARGET = "Churn"

categorical_cols = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

numerical_cols = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "num_addons", "charge_per_addon",
]

X = df[categorical_cols + numerical_cols]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# ---------------------------------------------------------------
# 3. Preprocessing pipeline
# ---------------------------------------------------------------
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, numerical_cols),
    ("cat", cat_transformer, categorical_cols),
])

# ---------------------------------------------------------------
# 4. Final tuned XGBoost pipeline (best params from GridSearchCV)
#    {'classifier__learning_rate': 0.05, 'classifier__max_depth': 3,
#     'classifier__n_estimators': 200}
# ---------------------------------------------------------------
scale = len(y[y == 0]) / len(y[y == 1])

xgb_pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=SEED)),
    ("classifier", XGBClassifier(
        scale_pos_weight=scale,
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        random_state=SEED,
        verbosity=0,
    )),
])

xgb_pipeline.fit(X_train, y_train)

# ---------------------------------------------------------------
# 5. Quick check on the test set
# ---------------------------------------------------------------
y_prob = xgb_pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------------
# 6. Save the fitted pipeline + threshold + column lists + SHAP
#    feature names together, so the app can explain predictions
# ---------------------------------------------------------------
feature_names = xgb_pipeline.named_steps["preprocessor"].get_feature_names_out()

artifact = {
    "model": xgb_pipeline,
    "threshold": THRESHOLD,
    "categorical_cols": categorical_cols,
    "numerical_cols": numerical_cols,
    "addon_cols": addon_cols,
    "feature_names": feature_names,
}

joblib.dump(artifact, MODEL_OUT_PATH)
print(f"\nSaved model to {MODEL_OUT_PATH}")