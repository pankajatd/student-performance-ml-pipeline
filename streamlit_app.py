"""
🎓 Student Exam Performance Indicator — Streamlit Cloud Web Application
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.set_page_config(
    page_title="Student Exam Performance Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .score-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: white;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎓 Student Exam Performance Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict student mathematics scores based on demographic and academic features using Machine Learning</div>', unsafe_allow_html=True)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["female", "male"])
        race_ethnicity = st.selectbox("Race / Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
        parental_level_of_education = st.selectbox(
            "Parental Level of Education",
            [
                "some high school",
                "high school",
                "some college",
                "associate's degree",
                "bachelor's degree",
                "master's degree"
            ]
        )
    
    with col2:
        lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
        test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"])
        reading_score = st.slider("Reading Score (0-100)", min_value=0, max_value=100, value=72)
        writing_score = st.slider("Writing Score (0-100)", min_value=0, max_value=100, value=74)
        
    submit_btn = st.form_submit_button("🚀 Predict Math Score", use_container_width=True, type="primary")

if submit_btn:
    try:
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=float(reading_score),
            writing_score=float(writing_score)
        )
        pred_df = data.get_data_as_data_frame()
        pipeline = PredictPipeline()
        preds = pipeline.predict(pred_df)
        score = round(float(preds[0]), 2)
        
        st.markdown(f"""
        <div class="score-card">
            <div style="font-size: 16px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9;">Predicted Mathematics Score</div>
            <div style="font-size: 54px; font-weight: 800; margin: 8px 0;">{score} / 100</div>
            <div style="font-size: 14px; opacity: 0.9;">Model inference complete using trained Scikit-Learn Regression Pipeline</div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
