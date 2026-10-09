
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Hide heading link icons
st.markdown(
    """
    <style>
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

BASE_DIR = Path(__file__).resolve().parent


# ---------------- LOAD SAVED FILES ----------------
@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "logistic_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    columns = joblib.load(BASE_DIR / "columns.pkl")
    return model, scaler, columns


# ---------------- SIDEBAR ----------------
st.sidebar.title("HeartCare AI")
st.sidebar.caption("Heart Disease Prediction")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Predict", "About Project"]
)

st.sidebar.divider()
st.sidebar.subheader("Model Information")
st.sidebar.write("Algorithm: Logistic Regression")
st.sidebar.write("Task: Classification")


# ---------------- MAIN HEADER ----------------
st.title("Heart Disease Prediction System")

st.write(
    "Enter the patient information to obtain a machine learning "
    "prediction based on the heart disease dataset."
)

st.info(
    "Educational project only. This prediction is not a medical "
    "diagnosis. Consult a qualified healthcare professional for "
    "medical assessment."
)

st.divider()


# ---------------- ABOUT PAGE ----------------
if page == "About Project":

    st.header("About the Project")

    st.write(
        "This project uses a trained Logistic Regression classifier "
        "to predict the heart disease label from patient health "
        "and examination features."
    )

    st.subheader("Input Features")

    st.markdown("""
    - Age
    - Sex
    - Chest Pain Type
    - Resting Blood Pressure
    - Cholesterol
    - Fasting Blood Sugar
    - Resting ECG
    - Maximum Heart Rate
    - Exercise-Induced Angina
    - Oldpeak
    - ST Slope
    """)

    st.subheader("Technologies Used")

    st.markdown("""
    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - Joblib
    """)

    st.warning(
        "The output represents a model classification, not a "
        "confirmed diagnosis or an individual's actual risk."
    )

    st.stop()


# ---------------- LOAD MODEL ----------------
try:
    model, scaler, feature_columns = load_model()

except Exception as error:
    st.error(f"Could not load the model files: {error}")
    st.write(
        "Keep logistic_model.pkl, scaler.pkl and columns.pkl "
        "in the same folder as app.py."
    )
    st.stop()


# ---------------- PATIENT INPUT FORM ----------------
st.header("Patient Information")
st.write("Complete the fields below.")

with st.form("heart_prediction_form"):

    st.subheader("1. Basic Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=45
        )

    with col2:
        sex = st.selectbox(
            "Sex",
            ["F", "M"],
            format_func=lambda x: {
                "F": "Female",
                "M": "Male"
            }[x]
        )

    with col3:
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ASY", "ATA", "NAP", "TA"],
            format_func=lambda x: {
                "ASY": "Asymptomatic (ASY)",
                "ATA": "Atypical Angina (ATA)",
                "NAP": "Non-Anginal Pain (NAP)",
                "TA": "Typical Angina (TA)"
            }[x]
        )

    st.divider()
    st.subheader("2. Clinical Measurements")

    col4, col5, col6 = st.columns(3)

    with col4:
        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=0,
            max_value=300,
            value=120
        )

    with col5:
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=1,
            max_value=1000,
            value=200
        )

    with col6:
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            [0, 1],
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
        )

    col7, col8, col9 = st.columns(3)

    with col7:
        resting_ecg = st.selectbox(
            "Resting ECG",
            ["LVH", "Normal", "ST"]
        )

    with col8:
        max_hr = st.number_input(
            "Maximum Heart Rate (bpm)",
            min_value=1,
            max_value=250,
            value=150
        )

    with col9:
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            ["N", "Y"],
            format_func=lambda x: {
                "N": "No",
                "Y": "Yes"
            }[x]
        )

    st.divider()
    st.subheader("3. Additional Measurements")

    col10, col11 = st.columns(2)

    with col10:
        oldpeak = st.number_input(
            "Oldpeak",
            min_value=-5.0,
            max_value=15.0,
            value=0.0,
            step=0.1,
            help="Enter the value used by the dataset."
        )

    with col11:
        st_slope = st.selectbox(
            "ST Slope",
            ["Down", "Flat", "Up"]
        )

    submitted = st.form_submit_button(
        "Predict Heart Disease",
        type="primary",
        use_container_width=True
    )


# ---------------- PREDICTION ----------------
if submitted:

    try:
        # Raw input names must match the training dataset.
        patient = pd.DataFrame([{
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "RestingECG": resting_ecg,
            "MaxHR": max_hr,
            "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope
        }])

        # Apply the same one-hot encoding approach as training.
        patient_encoded = pd.get_dummies(
            patient,
            dtype=int,
            drop_first=True
        )

        # Match the exact feature names and order used in training.
        patient_encoded = patient_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Use the saved training scaler.
        patient_scaled = scaler.transform(patient_encoded)

        prediction = int(model.predict(patient_scaled)[0])

        st.divider()
        st.header("Prediction Result")

        if prediction == 1:
            st.error(
                "The model classified this input as: "
                "Heart Disease Present"
            )
        else:
            st.success(
                "The model classified this input as: "
                "Heart Disease Not Detected"
            )

        st.subheader("Patient Summary")

        r1, r2, r3 = st.columns(3)

        r1.metric("Age", f"{age} years")
        r2.metric("Cholesterol", f"{cholesterol} mg/dL")
        r3.metric("Maximum Heart Rate", f"{max_hr} bpm")

        st.write(f"**Sex:** {sex}")
        st.write(f"**Chest Pain Type:** {chest_pain}")
        st.write(f"**Resting Blood Pressure:** {resting_bp} mm Hg")
        st.write(f"**Fasting Blood Sugar:** {fasting_bs}")
        st.write(f"**Resting ECG:** {resting_ecg}")
        st.write(f"**Exercise-Induced Angina:** {exercise_angina}")
        st.write(f"**Oldpeak:** {oldpeak}")
        st.write(f"**ST Slope:** {st_slope}")

        st.warning(
            "This is only a machine learning classification. "
            "It cannot confirm or rule out heart disease. "
            "Please consult a healthcare professional for medical advice."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# ---------------- FOOTER ----------------
st.divider()
st.caption("HeartCare AI | Machine Learning Project")