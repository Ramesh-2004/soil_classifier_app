import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model and label encoder
with open("model.pkl", "rb") as f:
    model, label_encoder = pickle.load(f)

# Function to plot plasticity chart
def plot_plasticity_chart(LL, PI):
    fig, ax = plt.subplots(figsize=(6, 5))

    # A-line and U-line
    x = list(range(0, 100))
    y_a = [0.73 * (i - 20) if i >= 20 else 0 for i in x]
    y_u = [0.9 * (i - 8) if i >= 8 else 0 for i in x]
    ax.plot(x, y_a, label='A-line', color='black')
    ax.plot(x, y_u, label='U-line', linestyle='--', color='red')

    # User input point
    ax.plot(LL, PI, 'bo', label='Your Input', markersize=10)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.set_xlabel("Liquid Limit (LL)")
    ax.set_ylabel("Plasticity Index (PI)")
    ax.set_title("Casagrande Plasticity Chart")
    ax.grid(True)
    ax.legend()

    return fig

# App UI
st.title("SOIL TYPE DETECTOR")
st.markdown("Enter the basic soil properties below:")

LL = st.number_input("Liquid Limit (LL)", min_value=0.0, step=0.1)
PL = st.number_input("Plastic Limit (PL)", min_value=0.0, step=0.1)
OMC = st.number_input("Optimum Moisture Content (OMC)", min_value=0.0, step=0.1)
MDD = st.number_input("Maximum Dry Density (MDD)", min_value=0.0, step=0.01)
PI = st.number_input("Plasticity Index (PI = LL - PL)", min_value=0.0, step=0.1)

if st.button("Predict Soil Type"):
    input_data = [[LL, PL, OMC, MDD, PI]]
    prediction = model.predict(input_data)
    soil_type = label_encoder.inverse_transform(prediction)[0]
    st.success(f"🌱 Predicted Soil Type: **{soil_type}**")

    # Optionally show confidence
    probs = model.predict_proba(input_data)
    st.info(f"Model Confidence: **{max(probs[0]) * 100:.2f}%**")

    # Show Casagrande Chart
    fig = plot_plasticity_chart(LL, PI)
    st.pyplot(fig)


