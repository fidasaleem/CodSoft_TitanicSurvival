import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import base64

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")


def set_bg(image_file):
    """Set background image to fully cover the app"""
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            height: 100vh;
        }}
        /* Make white transparent boxes readable */
        .result-box {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
            text-align: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background
set_bg(r"C:\Users\Fida\OneDrive\Documents\CodSoft_Titanic\titanic.jpeg")


# Load the trained model
rf_model = load("titanic_rf_model.pkl")
threshold = load("titanic_rf_threshold.pkl")


# Age to AgeBin mapping function
agebin_mapping = {
    'Child': 1,
    'Teen': 2,
    'Adult': 3,
    'MiddleAge': 4,
    'Senior': 5}

def get_age_bin(age):
    if age <= 12:
        return agebin_mapping['Child']
    elif age <= 19:
        return agebin_mapping['Teen']
    elif age <= 35:
        return agebin_mapping['Adult']
    elif age <= 60:
        return agebin_mapping['MiddleAge']
    else:
        return agebin_mapping['Senior']

# Deck mapping
deck_mapping = {
    'U': 0,  # Unknown
    'T': 1,
    'A': 2,
    'B': 3,
    'C': 4,
    'D': 5,
    'E': 6,
    'F': 7,
    'G': 8}

# FamilyGroup mapping
familygrp_mapping = {
    'Alone': 1,
    'Small': 2,
    'Large': 3}

# Streamlit App
st.title("Titanic Survival Predictor 🛳️")

st.write("""
Predict whether a passenger survived or not based on their characteristics.
""")

# User Input Form
with st.form("passenger_form"):
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    sibsp = st.number_input("Siblings/Spouses aboard (SibSp)", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Children aboard (Parch)", min_value=0, max_value=10, value=0)
    passenger_class = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
    fare = st.number_input("Fare", min_value=0.0, max_value=1000.0, value=32.0)
    family_category = st.selectbox("Family Category", ['Alone', 'Small', 'Large'])
    deck = st.selectbox("Deck", ['U','T','A','B','C','D','E','F','G'])
    embarked = st.selectbox("Embarked", ['S', 'Q', 'C'])

    submit = st.form_submit_button("Predict Survival")

if submit:
    # AgeBin
    age_bin = get_age_bin(age)

    # Deck
    deck_encoded = deck_mapping[deck]

    # Fare log
    fare_log = np.log1p(fare)

    pclass_1 = 1 if passenger_class == 1 else 0
    pclass_2 = 1 if passenger_class == 2 else 0
    pclass_3 = 1 if passenger_class == 3 else 0

    embarked_s = 1 if embarked == 'S' else 0
    embarked_q = 1 if embarked == 'Q' else 0
    embarked_c = 1 if embarked == 'C' else 0

    family_group = familygrp_mapping[family_category]

    is_alone = 1 if family_group == 1 else 0
    is_group = 1 if family_group > 1 else 0

    fare_median = 14.4542
    high_fare = 1 if fare > fare_median else 0

    has_cabin = 1 if deck != 'U' else 0

    input_data = pd.DataFrame({
        "Sex": [1 if sex == "female" else 0],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "AgeBin": [age_bin],
        "HasCabin": [has_cabin],
        "FamilyGroup": [family_group],
        "Deck": [deck_encoded],
        "IsAlone": [is_alone],
        "Fare_log": [fare_log],
        "HighFare": [high_fare],
        "IsGroup": [is_group],
        "Pclass_2": [pclass_2],
        "Pclass_3": [pclass_3],
        "Embarked_Q": [embarked_q],
        "Embarked_S": [embarked_s]
    })
    # Prediction
    prob = rf_model.predict_proba(input_data)[:, 1][0]
    survived = int(prob > threshold)

    result_text = "🎉 Survived!" if survived else "Did Not Survive"
    color = "green" if survived else "red"

    st.markdown(
        f"""
        <style>
        .result-box {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 10px 20px;
            border-radius: 10px;
            box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.2);
            text-align: center;
            width: 300px;
            margin: 20px auto;
        }}
        .result-box p {{
            color: #333;
            font-size: 14px;
            margin: 5px 0;
        }}
        .result-box h3 {{
            font-size: 20px;
            font-weight: bold;
            color: {color};
            margin-bottom: 10px;
        }}
        </style>

        <div class="result-box">
            <h3>{result_text}</h3>
            <p><b>Predicted Probability:</b> {prob:.2f}</p>
            <p><b>Threshold:</b> {threshold:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


