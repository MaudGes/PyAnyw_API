from flask import Flask, request, jsonify
import numpy as np
import mlflow.sklearn

app = Flask(__name__)

# Load the MLflow model
model = mlflow.sklearn.load_model("/home/MaudGes/mlflow_model")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    try:
        # Check if the input exists and has exactly 11 values
        if 'input' not in data or len(data['input']) != 11:
            return jsonify({'error': 'Invalid input. Expected an array of 11 values.'}), 400
        
        # Convert boolean values to 0 (False) or 1 (True)
        input_data = [1 if isinstance(x, bool) and x else 0 if isinstance(x, bool) else x for x in data['input']]
        
        # Transform the input into a NumPy array with the required shape (1, 11)
        input_data = np.array(input_data).reshape(1, -1)
        
        # Perform prediction
        prediction = model.predict(input_data)
        
        return jsonify({'prediction': int(prediction[0])})  # Convert prediction to int for JSON response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
