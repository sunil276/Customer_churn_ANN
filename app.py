import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load the trained model and preprocessors
model = load_model('model.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('one_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# App layout
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("🌟 Customer Churn Prediction")

# Sidebar: Collapsible sections for inputs
with st.sidebar:
    st.header("Customer Details")
    st.write("Enter the customer details below:")

    with st.expander("Personal Details", expanded=True):
        geography = st.selectbox("Geography", one_hot_encoder['Geography'].categories_[0])
        gender = st.selectbox("Gender", label_encoder['Gender'].classes_)
        age = st.slider("Age", 18, 92, step=1)
        tenure = st.slider("Tenure", 0, 10, step=1)

    with st.expander("Financial Details", expanded=False):
        credit_score = st.number_input("Credit Score", step=1.0, value=600.0)
        balance = st.number_input("Balance", step=100.0, value=0.0)
        estimated_salary = st.number_input("Estimated Salary", step=100.0, value=50000.0)

    with st.expander("Account Details", expanded=False):
        num_of_products = st.slider("Number of Products", 1, 4, step=1)
        has_cr_card = st.radio("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active_member = st.radio("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

# Prepare input data
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary],
})

# Preprocess data
input_data["Gender"] = label_encoder["Gender"].transform(input_data["Gender"])
encoded = one_hot_encoder["Geography"].transform(input_data[["Geography"]])
encoded_df = pd.DataFrame(encoded.toarray(), columns=one_hot_encoder["Geography"].get_feature_names_out(["Geography"]))
input_data = pd.concat([input_data.drop("Geography", axis=1), encoded_df], axis=1)
input_scaled = scaler.transform(input_data)

# Prediction
with st.spinner("Predicting..."):
    prediction_prob = model.predict(input_scaled)[0][0]

# Display results and confidence graph side by side
col1, col2 = st.columns([1, 1.5])
with col1:
    st.subheader("Prediction Results")
    st.metric("Churn Probability", f"{prediction_prob:.2%}")
    if prediction_prob > 0.5:
        st.error("⚠️ Likely to churn")
    else:
        st.success("✅ Not likely to churn")

with col2:
    st.subheader("Confidence Chart")
    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Adjusted size for better fit
    ax.bar(["Not Churn", "Churn"], [1 - prediction_prob, prediction_prob], color=["green", "red"])
    ax.set_title("Confidence Levels")
    ax.set_ylabel("Probability")
    st.pyplot(fig)
