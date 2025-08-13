import gradio as gr
import numpy as np
import pickle
import os

models = {
    "Random Forest": pickle.load(open("rf_heart_model.pkl", "rb")),
    "SVM": pickle.load(open("svm_heart_model.pkl", "rb")),
    "XGBoost": pickle.load(open("xgb_heart_model.pkl", "rb"))
}

scaler = pickle.load(open("scaler.pkl", "rb"))

def predict(model_name, age, anaemia, cpk, diabetes, ef, hbp, platelets,
            creatinine, sodium, sex, smoking, time):

    features = np.array([[age, anaemia, cpk, diabetes, ef, hbp, platelets,
                          creatinine, sodium, sex, smoking, time]])
    

    features_scaled = scaler.transform(features)

    model = models[model_name]
    prediction = model.predict(features_scaled)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(features_scaled)[0][1] * 100
        if prediction == 1:
            return f"❌ High Risk of Heart Failure ({prob:.2f}%)"
        else:
            return f"✅ Low Risk ({prob:.2f}%)"
    else:
        return "❌ High Risk" if prediction == 1 else "✅ Low Risk"

interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(["Random Forest", "SVM", "XGBoost"], label="Select Model"),
        gr.Number(label="Age (years)"),
        gr.Radio([0, 1], label="Anaemia (1 = Yes, 0 = No)"),
        gr.Number(label="Creatinine Phosphokinase (CPK level)"),
        gr.Radio([0, 1], label="Diabetes (1 = Yes, 0 = No)"),
        gr.Number(label="Ejection Fraction (%)"),
        gr.Radio([0, 1], label="High Blood Pressure (1 = Yes, 0 = No)"),
        gr.Number(label="Platelets"),
        gr.Number(label="Serum Creatinine"),
        gr.Number(label="Serum Sodium"),
        gr.Radio([0, 1], label="Sex (1 = Male, 0 = Female)"),
        gr.Radio([0, 1], label="Smoking (1 = Yes, 0 = No)"),
        gr.Number(label="Follow-up Time (days)")
    ],
    outputs=gr.Text(label="Prediction Result"),
    title="🫀 Heart Failure Prediction",
    description="Enter patient medical data and select a trained model to predict the risk of heart failure."
)

try:
    interface.launch(share=True)  
except:
    print("❌ Could not share publicly. Running locally only.")
    interface.launch(share=False)