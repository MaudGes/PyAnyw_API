from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd  # Import Pandas
import mlflow.sklearn
import sys

app = Flask(__name__)

# Load the MLflow model
model = mlflow.sklearn.load_model("/home/MaudGes/mysite/mlflow_model")

# Define the feature names expected by the model
FEATURE_NAMES = [
    "EXT_SOURCE_3", "EXT_SOURCE_2", "NAME_EDUCATION_TYPE_Higher education",
    "CODE_GENDER", "NAME_EDUCATION_TYPE_Secondary / secondary special",
    "FLAG_DOCUMENT_3", "AMT_REQ_CREDIT_BUREAU_HOUR", "REGION_RATING_CLIENT",
    "EXT_SOURCE_1", "NAME_INCOME_TYPE_Working", "FLAG_EMP_PHONE"
]

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None  # Default value

    if request.method == 'POST':
        try:
            # Retrieve form data
            input_data = [
                float(request.form['EXT_SOURCE_3']),
                float(request.form['EXT_SOURCE_2']),
                bool(request.form.get('NAME_EDUCATION_TYPE_Higher education')),
                int(request.form['CODE_GENDER']),
                bool(request.form.get('NAME_EDUCATION_TYPE_Secondary / secondary special')),
                int(request.form['FLAG_DOCUMENT_3']),
                float(request.form['AMT_REQ_CREDIT_BUREAU_HOUR']),
                int(request.form['REGION_RATING_CLIENT']),
                float(request.form['EXT_SOURCE_1']),
                bool(request.form.get('NAME_INCOME_TYPE_Working')),
                int(request.form['FLAG_EMP_PHONE'])
            ]

            # Convert input to DataFrame with feature names
            input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)

            # Debugging: Print input data
            print("✅ Received Data:", input_df)
            import sys
            sys.stdout.flush()  # Ensure logs are immediately written

            # Check if model exists
            if model is None:
                print("❌ ERROR: Model is None!")
                sys.stdout.flush()
                return render_template('index.html', error="Model is None.")

            # Check if input is valid before prediction
            print("🟢 Predicting...")
            sys.stdout.flush()
            
            prediction = model.predict(input_df)[0]  # Ensure this executes

            # Debugging: Print prediction
            print("✅ Prediction:", prediction)
            sys.stdout.flush()

        except Exception as e:
            print(f"❌ Exception Occurred: {e}")
            sys.stdout.flush()
            return render_template('index.html', error=str(e))

    print("🚀 Sending to template:", prediction)  # Debugging
    sys.stdout.flush()
    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)