import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page config
st.set_page_config(page_title="Dynamic Pricing Predictor", page_icon="💸", layout="centered")

# Title
st.title("💸 Dynamic Pricing Strategy Predictor")
st.markdown("Predict the **final optimal price** using your custom trained model.")

# Load model safely
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Input fields
st.header("Enter Input Features")

feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

# Predict button
if st.button("Predict Price"):
    try:
        input_data = np.array([[feature1, feature2, feature3]])
        prediction = model.predict(input_data)
        st.success(f"Predicted Price: ₹ {prediction[0]:,.2f}")
    except Exception as e:
        st.error("Prediction failed. Please check your inputs and model.")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Your Name")
