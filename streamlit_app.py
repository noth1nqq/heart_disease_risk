import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Загружаем модель
model = joblib.load('heart_disease_model1.pkl')

st.title("💓Heart Disease Prediction App")
st.write("Enter patient details below to get disease probability.")

age = st.slider("Age", 18, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG results", [0, 1, 2])
thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
thal = st.selectbox(f"Thalassemia:  \n3: no defect, 6: fixed defect, 7: reversible defect", [3, 6, 7])

# Преобразуем пол
sex = 1 if sex == "Male" else 0

# Создаем DataFrame с теми же колонками, что и на обучении
input_data = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}])



if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    if prediction == 1:
        st.error(f"High risk of heart disease: {probability:.2%}")
    else:
        st.success(f"✅ Low risk of heart disease: {probability:.2%}")
