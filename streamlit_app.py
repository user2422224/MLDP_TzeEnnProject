import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Airbnb Nightly Price Predictor", page_icon="🏠", layout="wide")
st.title("🏠 Airbnb Nightly Price Predictor")
st.caption("Enter listing details to predict an estimated nightly price using a trained regression model.")

MODEL_PATH = "model.pkl"
FEATURES_PATH = "feature_list.pkl"

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("model.pkl not found. Export it from your notebook using joblib.dump(...).")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError("feature_list.pkl not found. Export it from your notebook using joblib.dump(...).")
    model = joblib.load(MODEL_PATH)
    feature_list = joblib.load(FEATURES_PATH)
    return model, feature_list

try:
    model, feature_list = load_artifacts()
except Exception as e:
    st.error(str(e))
    st.stop()

st.subheader("Listing Details (Inputs)")

c1, c2 = st.columns(2)

with c1:
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=1, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=20, value=1, step=1)
    beds = st.number_input("Beds", min_value=0, max_value=50, value=1, step=1)
    toiles = st.number_input("Toilets", min_value=0, max_value=20, value=1, step=1)

with c2:
    guests = st.number_input("Guests (Capacity)", min_value=1, max_value=50, value=2, step=1)
    reviews = st.number_input("Number of Reviews", min_value=0, max_value=20000, value=10, step=1)
    rating = st.slider("Rating", min_value=0.0, max_value=5.0, value=4.5, step=0.1)

predict_btn = st.button("Predict Price", type="primary")

if predict_btn:
    # Build input row with the SAME feature names as training
    row = {
        "bathrooms": int(bathrooms),
        "beds": int(beds),
        "guests": int(guests),
        "toiles": int(toiles),
        "bedrooms": int(bedrooms),
        "reviews": int(reviews),
        "rating": float(rating),
    }

    # Ensure the DataFrame matches training schema exactly (same columns + order)
    X_input = pd.DataFrame([row])
    X_input = X_input.reindex(columns=feature_list)

    try:
        pred = model.predict(X_input)
        pred_value = float(np.squeeze(pred))
        st.success(f"Predicted Nightly Price: ${pred_value:,.2f}")
        st.write("Inputs:", row)
    except Exception as e:
        st.error("Prediction failed due to feature mismatch or model error.")
        st.code(str(e))
