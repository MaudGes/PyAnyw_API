from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib  # Import joblib to load the pipeline

app = Flask(__name__)

# Load the trained pipeline
pipeline = joblib.load('/home/MaudGes/mysite/pipeline_clients_traintest_4.joblib')

# Define the expected feature names
FEATURE_NAMES = [
    "EXT_SOURCE_3",
    "EXT_SOURCE_2",
    "NAME_EDUCATION_TYPE_Higher education",
    "NAME_INCOME_TYPE_Working",
    "NAME_EDUCATION_TYPE_Secondary / secondary special",
    "CODE_GENDER",
    "NAME_CONTRACT_TYPE_Cash loans",
    "REGION_RATING_CLIENT",
    "FLAG_DOCUMENT_3"
]

# Optimal threshold for final classification
OPTIMAL_THRESHOLD = 0.15  

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None  # Default value
    probability = None  # Default probability

    if request.method == 'POST':
        try:
            # Retrieve form data and convert to correct types
            input_data = [
                float(request.form['EXT_SOURCE_3']),
                float(request.form['EXT_SOURCE_2']),
                1 if request.form.get('NAME_EDUCATION_TYPE_Higher education') == 'on' else 0,
                1 if request.form.get('NAME_INCOME_TYPE_Working') == 'on' else 0,
                1 if request.form.get('NAME_EDUCATION_TYPE_Secondary / secondary special') == 'on' else 0,
                int(request.form['CODE_GENDER']),
                1 if request.form.get('NAME_CONTRACT_TYPE_Cash loans') == 'on' else 0,
                int(request.form['REGION_RATING_CLIENT']),
                int(request.form['FLAG_DOCUMENT_3'])
            ]

            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)

            print("✅ Received Data:", input_df)

            if pipeline is None:
                print("❌ ERROR: Pipeline is None!")
                return render_template('index.html', error="Pipeline not loaded.")

            print("🟢 Predicting...")

            # Get probability of repayment
            probability = pipeline.predict_proba(input_df)[:, 1][0]

            # Apply threshold to determine final classification
            prediction = int(probability >= OPTIMAL_THRESHOLD)

            print(f"✅ Probability of repayment: {probability:.2f}, Final prediction: {prediction}")

        except Exception as e:
            print(f"❌ Exception Occurred: {e}")
            return render_template('index.html', error=str(e))

    return render_template('index.html', probability=probability, prediction=prediction)