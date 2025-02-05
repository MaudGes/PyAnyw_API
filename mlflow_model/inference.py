import os
import pickle
import xgboost as xgb
import json
import numpy as np
import pandas as pd
import cloudpickle
from sagemaker_inference import content_types, response_types

def model_fn(model_dir):
    """Load the model from the model_dir (where SageMaker places the model)."""
    model_path = os.path.join(model_dir, 'model.pkl')
    # Load the model using cloudpickle or pickle
    with open(model_path, 'rb') as f:
        model = cloudpickle.load(f)  # Using cloudpickle to ensure compatibility with MLflow models
    return model

def input_fn(request_body, request_content_type):
    """Parse the input data from the request."""
    if request_content_type == content_types.JSON:
        # Parse JSON request body
        data = json.loads(request_body)
        # Convert it to a Pandas DataFrame (if your model expects a DataFrame input)
        return pd.DataFrame(data)
    elif request_content_type == content_types.CSV:
        # Parse CSV request body
        data = request_body.strip().split('\n')
        data = [list(map(float, row.split(','))) for row in data]
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported content type {request_content_type}")

def predict_fn(input_data, model):
    """Use the loaded model to make a prediction."""
    # Convert input to DMatrix for XGBoost prediction
    dmatrix = xgb.DMatrix(input_data)
    prediction = model.predict(dmatrix)
    return prediction

def output_fn(prediction, response_content_type):
    """Format the output."""
    if response_content_type == response_types.JSON:
        return json.dumps(prediction.tolist())  # Convert prediction to list and return as JSON
    elif response_content_type == response_types.CSV:
        return "\n".join(map(str, prediction))  # Return CSV formatted prediction
    else:
        raise ValueError(f"Unsupported response content type {response_content_type}")
