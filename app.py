from dotenv import load_dotenv
import os
import logging
import traceback
from flask import Flask, request, render_template, redirect, url_for
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Load environment variables
load_dotenv()

print("DEBUG from .env:", os.getenv("DEBUG"))

# Log environment variables
for k, v in os.environ.items():
    if "DEBUG" in k:
        print(f"{k} = {v}")

# Set up logging
log_file_path = os.path.join(os.getcwd(), "app_errors.log")
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=log_file_path, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logging.debug("Logging initialized successfully")

# Initialize Flask app
app = Flask(__name__)
debug_mode = os.getenv("DEBUG")

# Index route - redirecting to predictdata route
@app.route('/')
def index():
    return redirect(url_for('predict_datapoint'))

# Predict data route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    try:
        logging.debug(f"Received request at /predictdata method: {request.method}")
        if request.method == 'GET':
            logging.debug("Received GET request at /predictdata")
           
            print("Template path:", os.path.abspath("templates/home.html"))
            print("Exists?", os.path.exists("templates/home.html"))
            return render_template('home.html')

        # Handle POST request
        logging.debug("Received POST request at /predictdata")

        # Extract form data
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        logging.debug(f"Form data: {data.__dict__}")

        # Convert data to DataFrame and make prediction
        pred_df = data.get_data_as_data_frame()
        logging.debug(f"Prediction DataFrame: {pred_df.to_dict()}")

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        logging.debug(f"Prediction: {results}")

        return render_template('home.html', results=results[0])

    except Exception as e:
        error_trace = traceback.format_exc()
        logging.error(f"Exception in predict_datapoint: {str(e)}\n{error_trace}")
        return render_template('home.html', results="Prediction failed due to server error.")

# Print registered Flask routes (for debugging)
with app.test_request_context():
    print("Flask routes:")
    print(app.url_map)

if __name__ == "__main__":
    print("Flask app is starting...")
    app.run(host="0.0.0.0", debug=(debug_mode == "True"))