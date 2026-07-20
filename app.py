import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ==========================================================
# Load Dataset and Model
# ==========================================================

df = pd.read_csv("california_housing_streamlit.csv")
model = joblib.load("random_forest_model.pkl")

# ==========================================================
# Title
# ==========================================================

st.title("🏠 California House Price Prediction")

st.markdown("""
This application predicts **California house prices** using a trained **Random Forest Regression** model.

Adjust the feature values from the sidebar to get a prediction.
""")

# ==========================================================
# Sidebar Inputs
# ==========================================================

st.sidebar.header("Input Housing Features")

median_income = st.sidebar.slider(
    "Median Income",
    float(df["median_income"].min()),
    float(df["median_income"].max()),
    float(df["median_income"].mean())
)

housing_median_age = st.sidebar.slider(
    "Housing Median Age",
    int(df["housing_median_age"].min()),
    int(df["housing_median_age"].max()),
    int(df["housing_median_age"].median())
)

latitude = st.sidebar.slider(
    "Latitude",
    float(df["latitude"].min()),
    float(df["latitude"].max()),
    float(df["latitude"].mean())
)

longitude = st.sidebar.slider(
    "Longitude",
    float(df["longitude"].min()),
    float(df["longitude"].max()),
    float(df["longitude"].mean())
)

households = st.sidebar.slider(
    "Households",
    int(df["households"].min()),
    int(df["households"].max()),
    int(df["households"].median())
)

# ==========================================================
# Prediction
# ==========================================================

features = np.array([[
    median_income,
    housing_median_age,
    latitude,
    longitude,
    households
]])

prediction = model.predict(features)[0]

# ==========================================================
# Prediction Result
# ==========================================================

st.metric(
    label="🏡 Predicted House Price",
    value=f"${prediction:,.0f}"
)

# ==========================================================
# Display User Inputs
# ==========================================================

st.subheader("Selected Feature Values")

input_df = pd.DataFrame({
    "Feature": [
        "Median Income",
        "Housing Median Age",
        "Latitude",
        "Longitude",
        "Households"
    ],
    "Value": [
        median_income,
        housing_median_age,
        latitude,
        longitude,
        households
    ]
})

st.dataframe(input_df, use_container_width=True)

# ==========================================================
# Charts
# ==========================================================

col1, col2 = st.columns(2)

# ----------------------------------------------------------
# Histogram
# ----------------------------------------------------------

with col1:

    st.subheader("Median Income Distribution")

    fig = px.histogram(
        df,
        x="median_income",
        nbins=30,
        color_discrete_sequence=["royalblue"]
    )

    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# Scatter Plot
# ----------------------------------------------------------

with col2:

    st.subheader("Median Income vs House Price")

    fig = px.scatter(
        df,
        x="median_income",
        y="median_house_value",
        opacity=0.4,
        trendline="ols"
    )

    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# Correlation Heatmap
# ==========================================================

st.subheader("Correlation Heatmap")

corr = df.corr(numeric_only=True)

fig = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    aspect="auto"
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# Feature Importance
# ==========================================================

st.subheader("Feature Importance")

importance = pd.DataFrame({
    "Feature": [
        "Median Income",
        "Housing Median Age",
        "Latitude",
        "Longitude",
        "Households"
    ],
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=True
)

fig = px.bar(
    importance,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale="viridis"
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# Dataset Preview
# ==========================================================

st.subheader("Dataset Preview")

st.dataframe(df.head(10), use_container_width=True)

# ==========================================================
# Footer
# ==========================================================

st.markdown("---")

st.markdown(
"""
**Machine Learning Model:** Random Forest Regressor

**Dataset:** California Housing Prices
"""
)