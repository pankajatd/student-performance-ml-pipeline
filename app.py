import os
import sys

# Dual support: If launched by Streamlit (e.g., Streamlit Cloud), run the Streamlit UI
try:
    import streamlit as st
    if st.runtime.exists():
        import runpy
        runpy.run_path(os.path.join(os.path.dirname(__file__), "streamlit_app.py"), run_name="__main__")
        sys.exit(0)
except Exception:
    pass

# Otherwise run the Flask server
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import logging
import traceback
from flask import Flask, request, render_template, redirect, url_for
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Set up logging
log_file_path = os.path.join(os.getcwd(), "app_errors.log")
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=log_file_path, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# Initialize Flask app
app = Flask(__name__)
debug_mode = os.getenv("DEBUG")

@app.route('/')
def index():
    return redirect(url_for('predict_datapoint'))

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            return render_template('home.html')

        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])

    except Exception as e:
        error_trace = traceback.format_exc()
        logging.error(f"Exception in predict_datapoint: {str(e)}\n{error_trace}")
        return render_template('home.html', results="Prediction failed due to server error.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=(debug_mode == "True"))