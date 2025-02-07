from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd  # Import Pandas
import joblib  # Import joblib to load the pipeline

app = Flask(__name__)

# Load the joblib pipeline
pipeline = joblib.load('/home/MaudGes/mysite/pipeline_clients_traintest_2.joblib')

# Check if the pipeline performs transformations on the input data
print("✅ Checking pipeline steps:")
print(pipeline.named_steps)  # This will show all steps in the pipeline

# Define the feature names expected by the model
FEATURE_NAMES = [
    "EXT_SOURCE_3", "EXT_SOURCE_2", "NAME_EDUCATION_TYPE_Higher education",
    "CODE_GENDER", "NAME_EDUCATION_TYPE_Secondary / secondary special",
    "FLAG_DOCUMENT_3", "AMT_REQ_CREDIT_BUREAU_HOUR", "REGION_RATING_CLIENT",
    "EXT_SOURCE_1", "NAME_INCOME_TYPE_Working", "FLAG_EMP_PHONE"
]

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction request."""
    try:
        # Static input DataFrame (always use this on form submit)
        input_df = pd.DataFrame({
            'EXT_SOURCE_3': [0.1],
            'EXT_SOURCE_2': [0.2],
            'NAME_EDUCATION_TYPE_Higher education': [1],
            'CODE_GENDER': [0],  # Factorized (binary encoding)
            'NAME_EDUCATION_TYPE_Secondary / secondary special': [0],
            'FLAG_DOCUMENT_3': [1],
            'AMT_REQ_CREDIT_BUREAU_HOUR': [0.3],
            'REGION_RATING_CLIENT': [2],
            'EXT_SOURCE_1': [0.25],
            'NAME_INCOME_TYPE_Working': [1],
            'FLAG_EMP_PHONE': [1]
        })

        print("✅ Input DataFrame:")
        print(input_df)
        print(input_df.dtypes)
        print(input_df.columns)
        print(input_df.isnull().sum())  # Check for any missing values

        # Ensure pipeline is loaded
        if pipeline is None:
            print("❌ ERROR: Pipeline is None!")
            return render_template('index.html', error="Pipeline is None.")

        print(f"✅ Pipeline loaded: {type(pipeline)}")

        # Make prediction using the static DataFrame
        print("🟢 Predicting...")
        prediction = pipeline.predict(input_df)[0]  # Extract the first prediction value
        print("✅ Prediction:", prediction)

        # Return prediction to the template
        return render_template('index.html', prediction=prediction)

    except Exception as e:
        print(f"❌ ERROR during prediction: {e}")
        return render_template('index.html', error=str(e))