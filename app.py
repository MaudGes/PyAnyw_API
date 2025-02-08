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
    probability = None  # Default value

    if request.method == 'POST':
        try:
            # Step 1: Retrieve form data and convert booleans to 1 or 0
            input_data = [
                float(request.form['EXT_SOURCE_3']),
                float(request.form['EXT_SOURCE_2']),
                1 if request.form.get('NAME_EDUCATION_TYPE_Higher education') == 'on' else 0,  # Check if checked
                int(request.form['CODE_GENDER']),
                1 if request.form.get('NAME_EDUCATION_TYPE_Secondary / secondary special') == 'on' else 0,  # Check if checked
                int(request.form['FLAG_DOCUMENT_3']),
                1 if request.form.get('NAME_CONTRACT_TYPE_Cash loans') == 'on' else 0,  # Check if checked
                int(request.form['REGION_RATING_CLIENT']),
                float(request.form['EXT_SOURCE_1']),
                1 if request.form.get('NAME_INCOME_TYPE_Working') == 'on' else 0,  # Check if checked
            ]

            # Convert input to DataFrame with feature names
            input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)

            # Step 2: Model Loading Verification
            if pipeline is None:
                print("❌ ERROR: Pipeline is None!")
                return render_template('index.html', error="Pipeline is None.")
            else:
                print(f"✅ Pipeline loaded successfully: {type(pipeline)}")

            # Step 3: Make prediction using the pipeline
            prediction_prob = pipeline.predict_proba(input_df)[0][1]  # Get the probability for class 1 (non-remboursement)
            prediction = 1 if prediction_prob >= 0.15 else 0  # Apply the custom threshold (0.15)

            # Step 4: Debugging - Print prediction and probability
            print("✅ Prediction Probability:", prediction_prob)
            print("✅ Final Prediction:", prediction)

            probability = round(prediction_prob, 2)  # Round the probability to 2 decimal places for display

        except Exception as e:
            # Step 5: Exception Handling and Error Logs
            print(f"❌ Exception Occurred: {e}")
            return render_template('index.html', error=str(e))

    # Debugging: Log sending to template
    print("🚀 Sending to template:", prediction, probability)  # Debugging

    return render_template('index.html', prediction=prediction, probability=probability)