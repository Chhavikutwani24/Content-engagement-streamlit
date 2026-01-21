import streamlit as st
import pandas as pd
import joblib
import xgboost  # MUST be imported before loading model

st.set_page_config(page_title="Content Engagement Predictor")

st.title("📊 Content Engagement Prediction")
st.write("Predict whether a social media post will achieve high engagement")

# -----------------------------
# Load artifacts
# -----------------------------
model = joblib.load("content_engagement_model.pkl")
feature_list = joblib.load("feature_list.pkl")

# -----------------------------
# User Inputs
# -----------------------------
likes = st.number_input("Likes", min_value=0, value=100)
shares = st.number_input("Shares", min_value=0, value=10)
comments = st.number_input("Comments", min_value=0, value=5)
views = st.number_input("Views", min_value=1, value=1000)

# -----------------------------
# Feature Engineering (same logic)
# -----------------------------
total_engagement = likes + shares + comments
engagement_rate = total_engagement / (views + 1)
share_amplification = shares / (likes + 1)
comment_depth = comments / (likes + 1)
content_fatigue = views / (total_engagement + 1)

# -----------------------------
# IMPORTANT FIX:
# Create full feature frame FIRST
# -----------------------------
input_df = pd.DataFrame(
    data=[[0] * len(feature_list)],
    columns=feature_list
)

# Now overwrite known features
feature_updates = {
    "likes": likes,
    "shares": shares,
    "comments": comments,
    "views": views,
    "TotalEngagement": total_engagement,
    "EngagementRate": engagement_rate,
    "ShareAmplification": share_amplification,
    "CommentDepth": comment_depth,
    "ContentFatigue": content_fatigue
}

for col, val in feature_updates.items():
    if col in input_df.columns:
        input_df[col] = val

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Engagement"):
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success(f"🔥 High Engagement Likely (confidence={prob:.2f})")
    else:
        st.warning(f"⚠️ Low Engagement Risk (confidence={1 - prob:.2f})")

    risk_score = int((1 - prob) * 100)
    st.metric("Engagement Risk Score", risk_score)
