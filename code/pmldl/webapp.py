"""Streamlit frontend. All predictions are fetched from the separate HTTP API."""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="Iris classifier", page_icon="🌸", layout="centered")
st.title("Iris classifier")
st.write("Enter four measurements in centimetres to predict an Iris species.")

with st.form("iris_prediction"):
    left, right = st.columns(2)
    with left:
        sepal_length = st.number_input("Sepal length (cm)", 0.1, 30.0, 5.1, 0.1)
        petal_length = st.number_input("Petal length (cm)", 0.1, 30.0, 1.4, 0.1)
    with right:
        sepal_width = st.number_input("Sepal width (cm)", 0.1, 30.0, 3.5, 0.1)
        petal_width = st.number_input("Petal width (cm)", 0.1, 30.0, 0.2, 0.1)
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    st.session_state.pop("prediction", None)  # Do not display stale results on error.
    try:
        with st.spinner("Requesting a prediction from the API…"):
            response = requests.post(API_URL + "/predict", json={
                "sepal_length": sepal_length, "sepal_width": sepal_width,
                "petal_length": petal_length, "petal_width": petal_width,
            }, timeout=(3, 15))
            response.raise_for_status()
            prediction = response.json()
            if not {"species", "probabilities", "model_run_id"} <= prediction.keys():
                raise ValueError("Invalid API response")
            st.session_state["prediction"] = prediction
    except (requests.RequestException, ValueError) as error:
        st.error(f"Prediction failed. Check that the API is running and retry. {error}")

if "prediction" in st.session_state:
    prediction = st.session_state["prediction"]
    st.success(f"Predicted species: {prediction['species']}")
    probabilities = prediction["probabilities"]
    st.bar_chart({"species": list(probabilities), "probability": list(probabilities.values())},
                 x="species", y="probability")
    st.caption(f"Model run: {prediction['model_run_id']}")

with st.expander("Current model and test metrics"):
    try:
        response = requests.get(API_URL + "/model", timeout=(2, 5))
        response.raise_for_status()
        metadata = response.json()
        st.json({"run_id": metadata["run_id"], "trained_at": metadata["trained_at"],
                 "test_metrics": metadata["metrics"]})
    except (requests.RequestException, ValueError, KeyError):
        st.info("Model information is temporarily unavailable (the API may be redeploying).")

st.caption("Educational MLOps demo · DVC → MLflow → FastAPI + Streamlit")
