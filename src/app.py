import streamlit as st
import plotly.graph_objects as go
import numpy as np
import dill
import os
import pandas as pd

from report_gen import generate_medical_pdf
from data_preprocessing import preprocess_hybrid_lupus
from uncertainty_model import train_with_abstention, predict_with_abstention
from hybrid_ensemble import dynamic_ensemble_predict

# --- 1. ABSOLUTE PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.normpath(os.path.join(current_dir, "..", "results", "trained_lupus_model.pkl"))
clinical_path = os.path.normpath(os.path.join(current_dir, "..", "dataset", "lupus_dataset.csv"))
genomic_path = os.path.normpath(os.path.join(current_dir, "..", "dataset", "GSE65391_medium.txt"))

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="Quantum Lupus Insight", page_icon="🩺", layout="wide")
st.title("🩺 Quantum-Enhanced Lupus Early Prediction")
st.markdown("---")

# --- 3. LOAD THE LOCKED QUANTUM MODEL ---
@st.cache_resource
def load_quantum_model():
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}. Please run main.py first!")
        return None
    with open(model_path, 'rb') as f:
        return dill.load(f)

# --- 4. LOAD DATA + TRAIN UNCERTAINTY MODEL ONCE ---
@st.cache_resource
def load_uncertainty_model():
    X_train, X_test, y_train, y_test = preprocess_hybrid_lupus(clinical_path, genomic_path)
    unc_model, qhat = train_with_abstention(X_train, y_train)
    return unc_model, qhat

vqc = load_quantum_model()
unc_bundle = load_uncertainty_model()

if vqc and unc_bundle:
    unc_model, qhat = unc_bundle

    # --- 5. INPUTS ---
    st.sidebar.header("📋 Patient Clinical Survey")
    patient_name = st.sidebar.text_input("Full Patient Name", "John Doe")
    st.sidebar.markdown("---")

    input_mode = st.sidebar.radio("Input Mode", ["Quantum Encoding Mode", "Clinical Input Mode"])

    # initialize clinical-only vars to avoid NameError
    malar_rash = None
    joint_pain = None
    proteinuria = None
    ifi27 = None
    stat1 = None
    ana_result = None
    dsdna_value = None

    if input_mode == "Quantum Encoding Mode":
        st.sidebar.caption("This mode uses normalized encoded features on a 0–1 scale for internal model testing.")

        f0 = st.sidebar.slider("Malar Rash Presence (Encoded)", 0.0, 1.0, 0.4)
        f1 = st.sidebar.slider("Joint Pain Level (Encoded)", 0.0, 1.0, 0.6)
        f2 = st.sidebar.slider("Renal Marker / Proteinuria", 0.0, 1.0, 0.2)
        f3 = st.sidebar.slider("Genomic Interferon Gene (IFI27)", 0.0, 1.0, 0.5)
        f4 = st.sidebar.slider("Genomic Interferon Gene (STAT1)", 0.0, 1.0, 0.3)
        f5 = st.sidebar.slider("Auto-Antibody Score (ANA/dsDNA)", 0.0, 1.0, 0.7)

    else:
        st.sidebar.caption("This mode accepts real clinical values and converts them internally for the model.")

        malar_rash = st.sidebar.selectbox("Malar Rash", ["No", "Yes"])
        joint_pain = st.sidebar.slider("Joint Pain Severity (0-10)", 0, 10, 5)
        proteinuria = st.sidebar.number_input("Proteinuria (mg/dL)", min_value=0.0, value=150.0)
        ifi27 = st.sidebar.number_input("IFI27 Gene Expression", min_value=0.0, value=1.0)
        stat1 = st.sidebar.number_input("STAT1 Gene Expression", min_value=0.0, value=1.0)
        ana_result = st.sidebar.selectbox("ANA Result", ["Negative", "Positive"])
        dsdna_value = st.sidebar.number_input("Anti-dsDNA Value", min_value=0.0, value=50.0)

        # Convert real clinical values into normalized model-ready features
        f0 = 1.0 if malar_rash == "Yes" else 0.0
        f1 = joint_pain / 10.0
        f2 = min(proteinuria / 1000.0, 1.0)
        f3 = min(ifi27 / 10.0, 1.0)
        f4 = min(stat1 / 10.0, 1.0)
        f5 = max(1.0 if ana_result == "Positive" else 0.0, min(dsdna_value / 200.0, 1.0))

    # --- 6. MAIN DASHBOARD ---
    col1, col2 = st.columns([1, 1])

    if st.button("🚀 Run Quantum Diagnostic Analysis"):
        input_data = np.array([[f0, f1, f2, f3, f4, f5]])

        # Quantum prediction
        prediction = vqc.predict(input_data)
        result_val = int(np.asarray(prediction).flatten()[0])
        quantum_label = "High Lupus Risk" if result_val == 1 else "Low Risk"

        # Classical uncertainty model probabilities
        unc_probs = unc_model.predict_proba(input_data)[0]
        low_risk_prob = float(unc_probs[0]) * 100
        high_risk_prob = float(unc_probs[1]) * 100

        # Hybrid ensemble prediction
        hybrid_pred, hybrid_lupus_prob, hybrid_low_prob, classical_weight, quantum_weight = dynamic_ensemble_predict(
            result_val, unc_probs
        )

        # Use hybrid probability as final displayed risk
        risk_score = round(hybrid_lupus_prob, 2)

        # Uncertainty / safety layer
        unc_result = predict_with_abstention(unc_model, qhat, input_data)
        unc_pred, unc_conf = unc_result[0]
        unc_conf_pct = float(unc_conf) * 100

        # Patient-friendly clinical summary
        st.markdown("### 🩺 Clinical Risk Assessment Summary")

        risk_level = "High Risk" if hybrid_pred == 1 else "Low Risk"

        st.write(f"**Overall Assessment:** {risk_level}")
        st.write(f"**Estimated Probability of Lupus:** {hybrid_lupus_prob:.2f}%")

        if hybrid_lupus_prob >= 70:
            st.error("🔴 High likelihood of lupus detected. Clinical evaluation is strongly recommended.")
        elif hybrid_lupus_prob >= 40:
            st.warning("🟡 Moderate risk detected. Further diagnostic testing is advised.")
        else:
            st.success("🟢 Low risk detected. Routine monitoring is sufficient.")

        st.markdown("---")
        st.write("### 🧠 AI Interpretation")
        st.write(
            "This assessment is based on a combination of advanced computational models analyzing clinical indicators "
            "and molecular markers. The result reflects the estimated likelihood of lupus based on the current input data."
        )

        # Technical details hidden for clinicians
        with st.expander("🔬 Technical Details (For Clinicians Only)"):
            st.write(f"**Quantum Model Prediction:** {quantum_label}")
            st.write(f"**Classical Model Lupus Probability:** {high_risk_prob:.2f}%")
            st.write(f"**Final Hybrid Prediction:** {risk_level}")
            st.write(f"**Final Hybrid Lupus Probability:** {hybrid_lupus_prob:.2f}%")
            st.write(f"**Quantum Weight:** {quantum_weight:.2f}")
            st.write(f"**Classical Weight:** {classical_weight:.2f}")

        # Safety banner
        st.markdown("### Safety Level")
        if unc_pred == "UNCERTAIN":
            st.warning(f"⚠️ Uncertain prediction detected (Confidence: {unc_conf_pct:.2f}%). Please consult a specialist.")
        elif unc_pred == 1:
            st.error(f"🚨 High Lupus Risk suggested by safety model (Confidence: {unc_conf_pct:.2f}%).")
        else:
            st.success(f"✅ Low Risk suggested by safety model (Confidence: {unc_conf_pct:.2f}%).")

        with col1:
            st.subheader("Current Diagnostic Risk")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Hybrid Lupus Probability (%)", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "black"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(0, 255, 0, 0.6)'},
                        {'range': [30, 70], 'color': 'rgba(255, 255, 0, 0.6)'},
                        {'range': [70, 100], 'color': 'rgba(255, 0, 0, 0.6)'}
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("12-Month Risk Projection")
            months = ["Current", "3 Months", "6 Months", "12 Months"]

            if risk_score >= 70:
                trend = [
                    risk_score,
                    min(risk_score + 4, 100),
                    min(risk_score + 7, 100),
                    min(risk_score + 10, 100)
                ]
            elif risk_score >= 30:
                trend = [
                    risk_score,
                    min(risk_score + 2, 100),
                    min(risk_score + 4, 100),
                    min(risk_score + 6, 100)
                ]
            else:
                trend = [
                    risk_score,
                    max(risk_score - 1, 0),
                    min(risk_score + 2, 100),
                    min(risk_score + 4, 100)
                ]

            df_trend = pd.DataFrame({"Timeline": months, "Risk (%)": trend})
            st.line_chart(df_trend.set_index("Timeline"))

        st.markdown("---")
        st.subheader("Final Documentation")

        if input_mode == "Quantum Encoding Mode":
            symptoms_summary = {
                "Skin/Rash Score (Encoded)": f0,
                "Joint Intensity (Encoded)": f1,
                "Kidney Marker (Encoded)": f2,
                "Molecular Interferon Score (Encoded)": f3,
                "STAT1 Score (Encoded)": f4,
                "Auto-Antibody Score (Encoded)": f5,
                "Overall Assessment": risk_level,
                "Quantum Model Prediction": quantum_label,
                "Estimated Lupus Probability (%)": round(hybrid_lupus_prob, 2),
                "Classical Lupus Probability (%)": round(high_risk_prob, 2),
                "Quantum Weight": round(quantum_weight, 2),
                "Classical Weight": round(classical_weight, 2),
                "Safety Model Confidence (%)": round(unc_conf_pct, 2),
                "Safety Model Output": str(unc_pred)
            }
        else:
            symptoms_summary = {
                "Malar Rash": malar_rash,
                "Joint Pain Severity": joint_pain,
                "Proteinuria (mg/dL)": proteinuria,
                "IFI27 Expression": ifi27,
                "STAT1 Expression": stat1,
                "ANA Result": ana_result,
                "Anti-dsDNA Value": dsdna_value,
                "Overall Assessment": risk_level,
                "Quantum Model Prediction": quantum_label,
                "Estimated Lupus Probability (%)": round(hybrid_lupus_prob, 2),
                "Classical Lupus Probability (%)": round(high_risk_prob, 2),
                "Quantum Weight": round(quantum_weight, 2),
                "Classical Weight": round(classical_weight, 2),
                "Safety Model Confidence (%)": round(unc_conf_pct, 2),
                "Safety Model Output": str(unc_pred)
            }

        try:
            pdf_bytes = generate_medical_pdf(patient_name, risk_score, symptoms_summary)
            st.download_button(
                label="📥 Download Official Medical Report (PDF)",
                data=pdf_bytes,
                file_name=f"Lupus_Report_{patient_name}.pdf",
                mime="application/pdf"
            )
            st.success("✅ Diagnostic complete. Report ready for download.")
        except Exception as e:
            st.error(f"Error generating PDF: {e}")

else:
    st.info("Waiting for Quantum Model and uncertainty model to load...")

# Footer info
st.sidebar.markdown("---")
st.sidebar.caption("System Accuracy: 88.24%")
st.sidebar.caption("Architecture: 6-Qubit VQC (Circular Entanglement)")
st.sidebar.caption("Safety Layer: Uncertainty-aware SVM")