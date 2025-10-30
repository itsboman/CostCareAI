import streamlit as st
import pandas as pd
import joblib
import base64
import os


# ==============================
# ✅ PAGE CONFIG
# ==============================
st.set_page_config(page_title="CostCareAI", page_icon="logo(2).png", layout="centered")

# ==============================
# ✅ CUSTOM STYLING (Dark Theme)
# ==============================
st.markdown("""
    <style>
        body {
            background-color: #0f172a;
            color: #f8fafc;
            font-family: 'Poppins', sans-serif;
        }
        .main {
            background-color: #0f172a;
            color: white;
        }
        .logo-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 40px;
            margin-bottom: 15px;
            animation: fadeIn 2s ease-in-out;
        }
        .logo-container img {
            width: 360px;
            height: auto;
            border-radius: 20px;
            box-shadow: 0px 0px 40px rgba(56, 189, 248, 0.5);
        }
        .title {
            text-align: center;
            font-size: 45px;
            font-weight: 800;
            color: #38bdf8;
            margin-top: 15px;
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #94a3b8;
            margin-bottom: 35px;
        }
        .stTextInput > div > div > input,
        .stNumberInput input,
        .stSelectbox select {
            background-color: #1e293b;
            color: white;
            border-radius: 10px;
            border: 1px solid #38bdf8;
        }
        .stButton button {
            background-color: #38bdf8;
            color: black;
            border-radius: 10px;
            border: none;
            font-size: 18px;
            font-weight: bold;
            padding: 10px 20px;
            transition: 0.3s;
        }
        .stButton button:hover {
            background-color: #0ea5e9;
            color: white;
            transform: scale(1.05);
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.9); }
            to { opacity: 1; transform: scale(1); }
        }
    </style>
""", unsafe_allow_html=True)

# ==============================
# ✅ LOAD MODEL
# ==============================
model_path = "model.pkl"
if not os.path.exists(model_path):
    st.error("❌ Model file 'model.pkl' not found! Please place it in the same folder as this script.")
else:
    model = joblib.load(model_path)

# ==============================
# ✅ LOGO + TITLE
# ==============================
logo_path = "logo(2).png"  # Ensure this file is in the same folder as app.py

if os.path.exists(logo_path):
    with open(logo_path, "rb") as image_file:
        encoded_logo = base64.b64encode(image_file.read()).decode()

    st.markdown(f"""
        <div class="logo-container">
            <img src="data:image/png;base64,{encoded_logo}">
        </div>
    """, unsafe_allow_html=True)
else:
    st.warning("⚠️ Logo not found. Make sure 'logo(2).png' is in the same folder as app.py")

st.markdown('<div class="title"> Hospital Bill Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict estimated medical expenses based on patient details</div>', unsafe_allow_html=True)

# ==============================
# ✅ USER INPUT SECTION
# ==============================
st.markdown("###  Enter Patient Details")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=1)
with col2:
    gender = st.selectbox("Gender", ["male", "female"])
    smoker = st.selectbox("Smoker?", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# ==============================
# ✅ PREDICTION
# ==============================
if st.button(" Predict Hospital Bill"):
    # Convert categorical inputs into numeric (same as your training encodings)
    sex_encoded = 1 if gender == "male" else 0
    smoker_encoded = 1 if smoker == "yes" else 0
    region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
    region_encoded = region_map[region]

    input_df = pd.DataFrame({
        "age": [age],
        "sex": [sex_encoded],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker_encoded],
        "region": [region_encoded]
    })

    try:
        prediction = model.predict(input_df)[0]
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #0ea5e9, #22d3ee);
                        padding: 25px; border-radius: 20px;
                        text-align: center; color: white;
                        font-size: 26px; font-weight: 700;
                        box-shadow: 0px 0px 30px rgba(56, 189, 248, 0.5);'>
                 Estimated Hospital Bill: ${prediction:,.2f}
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"⚠️ Something went wrong: {e}")
# 💾 Save model using joblib
joblib.dump(model, "model.pkl")
print("💾 Model saved as model.pkl")



