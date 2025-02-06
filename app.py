from flask import Flask, request, jsonify, render_template
import numpy as np
import mlflow.sklearn

app = Flask(__name__)

# Load the MLflow model
model = mlflow.sklearn.load_model("/home/MaudGes/mysite/mlflow_model")

# Home route with form
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None  # Default value for the result
    
    if request.method == 'POST':
        try:
            # Retrieve form data and convert to proper types
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

            # Convert input data into numpy array with shape (1, 11)
            input_data = np.array(input_data, dtype=np.float32).reshape(1, -1)

            # Perform prediction
            prediction = model.predict(input_data)[0]  # Extract prediction

        except Exception as e:
            return render_template('index.html', error=str(e))  # Show error message

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
