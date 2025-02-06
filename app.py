from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd  # Import Pandas
import joblib  # Import joblib to load the pipeline

app = Flask(__name__)

# Load the joblib pipeline
pipeline = joblib.load('/home/MaudGes/mysite/pipeline_clients_traintest.joblib')

# Define the feature names expected by the model
FEATURE_NAMES = [
    "EXT_SOURCE_3", "EXT_SOURCE_2", "NAME_EDUCATION_TYPE_Higher education",
    "CODE_GENDER", "NAME_EDUCATION_TYPE_Secondary / secondary special",
    "FLAG_DOCUMENT_3", "AMT_REQ_CREDIT_BUREAU_HOUR", "REGION_RATING_CLIENT",
    "EXT_SOURCE_1", "NAME_INCOME_TYPE_Working", "FLAG_EMP_PHONE"
]

@app.route('/', methods=['GET', 'POST'])
def home():
    # Set Pandas options to show full DataFrame
    import pandas as pd
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)        # Prevent line breaks
    pd.set_option('display.max_rows', None)     # Show all rows

    prediction = None  # Default value

    if request.method == 'POST':
        try:
            # Step 1: Retrieve form data and clean it
            input_data = [
                clean_input(request.form['EXT_SOURCE_3']),
                clean_input(request.form['EXT_SOURCE_2']),
                1 if request.form.get('NAME_EDUCATION_TYPE_Higher education') == 'on' else 0,  # Check if checked
                clean_input(request.form['CODE_GENDER']),
                1 if request.form.get('NAME_EDUCATION_TYPE_Secondary / secondary special') == 'on' else 0,  # Check if checked
                clean_input(request.form['FLAG_DOCUMENT_3']),
                clean_input(request.form['AMT_REQ_CREDIT_BUREAU_HOUR']),
                clean_input(request.form['REGION_RATING_CLIENT']),
                clean_input(request.form['EXT_SOURCE_1']),
                1 if request.form.get('NAME_INCOME_TYPE_Working') == 'on' else 0,  # Check if checked
                clean_input(request.form['FLAG_EMP_PHONE'])
            ]

            # Convert input to DataFrame with feature names
            input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)

            # Step 2: Debugging - Print input data
            print("✅ Received Data:")
            print(input_df)

            # Step 3: Model Loading Verification
            if pipeline is None:
                print("❌ ERROR: Pipeline is None!")
                return render_template('index.html', error="Pipeline is None.")
            else:
                print(f"✅ Pipeline loaded successfully: {type(pipeline)}")

            # Step 4: Check if input is valid before prediction
            print("🟢 Predicting...")

            # Step 5: Make prediction using the pipeline
            prediction = pipeline.predict(input_df)[0]  # Ensure this executes

            # Step 6: Debugging - Print prediction
            print("✅ Prediction:", prediction)

        except Exception as e:
            # Step 7: Exception Handling and Error Logs
            print(f"❌ Exception Occurred: {e}")
            return render_template('index.html', error=str(e))

    # Debugging: Log sending to template
    print("🚀 Sending to template:", prediction)  # Debugging

    return render_template('index.html', prediction=prediction)
