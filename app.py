import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="Flood Risk Prediction System",
    page_icon="🌊",
    layout="wide"
)

# ------------------------------------------------
# Load Model
# ------------------------------------------------
model = joblib.load("models/flood_risk_model.pkl")

risk_labels = ["Low", "Medium", "High"]

# ------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Risk Assessment", "Model Insights", "About"]
)

# ==============================================================
# PAGE 1 — RISK ASSESSMENT
# ==============================================================
if page == "Risk Assessment":

    st.title("Flood Risk Prediction System")
    st.markdown(
        "AI-driven hydrological risk classification engine using machine learning."
    )

    st.divider()

    st.subheader("Input Parameters")

    col1, col2 = st.columns(2)

    with col1:
        rainfall = st.number_input(
            "Rainfall (mm)", min_value=0.0, value=150.0
        )
        discharge = st.number_input(
            "River Discharge (m³/s)", min_value=0.0, value=2500.0
        )

    with col2:
        water_level = st.number_input(
            "Water Level (m)", min_value=0.0, value=5.0
        )
        elevation = st.number_input(
            "Elevation (m)", min_value=0.0, value=4400.0
        )

    historical = st.selectbox(
        "Historical Flood Indicator (0 = No, 1 = Yes)",
        [0, 1]
    )

    # Basic validation
    if rainfall > 1000:
        st.warning("Rainfall unusually high. Please verify input values.")

    if st.button("Run Risk Assessment"):

        input_data = np.array([[rainfall, discharge, water_level, elevation, historical]])

        with st.spinner("Processing hydrological indicators..."):
            prediction = model.predict(input_data)
            probabilities = model.predict_proba(input_data)[0]

        risk_level = prediction[0]

        st.divider()
        st.subheader("Risk Assessment Summary")

        # KPI Metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Predicted Risk Level", risk_labels[risk_level])

        with col2:
            st.metric(
                "Model Confidence",
                f"{np.max(probabilities) * 100:.2f}%"
            )

        with col3:
            st.metric(
                "Historical Flood Indicator",
                historical
            )

        st.divider()

        # Probability Distribution Chart
        st.subheader("Risk Probability Distribution")

        fig, ax = plt.subplots()

        bars = ax.bar(risk_labels, probabilities)

        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        ax.set_title("Predicted Risk Probabilities")

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom"
            )

        st.pyplot(fig)

        st.divider()

        st.markdown(
            """
            **Risk Level Interpretation**

            - **Low:** Minimal hydrological stress conditions  
            - **Medium:** Moderate flood potential  
            - **High:** Elevated flood likelihood; monitoring recommended  
            """
        )

# ==============================================================
# PAGE 2 — MODEL INSIGHTS
# ==============================================================
elif page == "Model Insights":

    st.title("Model Insights")

    st.subheader("Performance Metrics")

    st.metric("Test Accuracy", "98.8%")
    st.metric("Macro F1 Score", "0.98")

    st.divider()

    st.subheader("Feature Importance")

    importance = model.feature_importances_

    features = [
        "Rainfall (mm)",
        "River Discharge (m³/s)",
        "Water Level (m)",
        "Elevation (m)",
        "Historical Floods"
    ]

    fig, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance Score")

    st.pyplot(fig)

    st.info(
        """
        Model: XGBoost Classifier  
        Features: 5 Hydrological Indicators  
        Labeling Method: KMeans-derived risk categorization  
        Validation Strategy: Stratified Train-Test Split  
        """
    )

# ==============================================================
# PAGE 3 — ABOUT
# ==============================================================
else:

    st.title("About This Project")

    st.markdown(
        """
        ### Flood Risk Prediction System

        This project demonstrates an end-to-end machine learning workflow:

        1. Initial binary flood prediction showed limited predictive signal.
        2. Risk zones were derived using KMeans clustering.
        3. A supervised XGBoost classifier was trained to predict derived risk levels.
        4. The deployed system provides real-time risk assessment via Streamlit.

        ### Technology Stack

        - Python  
        - Scikit-learn  
        - XGBoost  
        - Streamlit  
        - Matplotlib  

        This system illustrates practical application of unsupervised + supervised learning
        for environmental risk modeling.
        """
    )

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.divider()
st.caption(
    "Flood Risk Prediction System v1.0 | Developed by Soumili Saha | ML Project 2026"
)