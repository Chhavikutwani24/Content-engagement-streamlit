# !pip install xgboost joblib

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier


# -----------------------------
# Load Dataset (URL)
# -----------------------------
url = "https://raw.githubusercontent.com/intern2grow/social-media-data-analysis/main/social_media_data.csv"
data = pd.read_csv(url)

print(data.shape)
print(data.head())


# -----------------------------
# Cleaning
# -----------------------------
data.dropna(inplace=True)


# -----------------------------
# Feature Engineering (NON-BASIC)
# -----------------------------

# Total engagement signal
data["TotalEngagement"] = (
    data["likes"] + data["shares"] + data["comments"]
)

# Engagement rate (normalized)
data["EngagementRate"] = (
    data["TotalEngagement"] / (data["views"] + 1)
)

# Share amplification (virality signal)
data["ShareAmplification"] = (
    data["shares"] / (data["likes"] + 1)
)

# Comment depth (discussion signal)
data["CommentDepth"] = (
    data["comments"] / (data["likes"] + 1)
)

# Content fatigue proxy
data["ContentFatigue"] = (
    data["views"] / (data["TotalEngagement"] + 1)
)


# -----------------------------
# Target Engineering
# -----------------------------
# High engagement = top 25% engagement rate
threshold = data["EngagementRate"].quantile(0.75)
data["HighEngagement"] = (
    data["EngagementRate"] >= threshold
).astype(int)


# -----------------------------
# Encoding
# -----------------------------
encoder = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = encoder.fit_transform(data[col])


# -----------------------------
# Features / Target
# -----------------------------
X = data.drop(columns=["HighEngagement"])
y = data["HighEngagement"]


# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# -----------------------------
# Model
# -----------------------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)


# -----------------------------
# Evaluation
# -----------------------------
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, preds))
print("ROC AUC:", roc_auc_score(y_test, probs))


# -----------------------------
# Save Model & Features
# -----------------------------
joblib.dump(model, "content_engagement_model.pkl")
joblib.dump(X.columns.tolist(), "feature_list.pkl")

print("✅ Saved content_engagement_model.pkl & feature_list.pkl")
