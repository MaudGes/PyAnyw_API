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

def clean_input(value):
    return str(value).replace("\n", "").replace("\r", "").strip()  # Remove newlines and extra spaces

@app.route('/', methods=['GET', 'POST'])
def home():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)

    prediction = None  # Default value

    if request.method == 'POST':  # When user clicks "Submit"
        try:
            # 🔹 Bypass form input and directly define the DataFrame
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

            # Check if pipeline is loaded
            if pipeline is None:
                print("❌ ERROR: Pipeline is None!")
                return render_template('index.html', error="Pipeline is None.")

            print("🟢 Predicting...")
            prediction = pipeline.predict(input_df)[0]  # Extract first value

            print("✅ Prediction:", prediction)

        except Exception as e:
            print(f"❌ ERROR during prediction: {e}")
            return render_template('index.html', error=str(e))

    # Debugging: Log sending to template
    print("🚀 Sending to template:", prediction)  # Debugging

    return render_template('index.html', prediction=prediction)