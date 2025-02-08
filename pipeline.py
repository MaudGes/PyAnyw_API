from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
import time
import gc
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import warnings

"""
Pipeline used to easily define the model preprocessing steps. The pipeline is then stored using joblib. 
The model's signature is also saved to be able to reuse it later while deploying the model.
The model itself is then stored along with other files in the 'mlflow_model' folder

The selected hyperparaters for xgboost come from previous testing and results stored in mlflow (see mlruns folder)
"""

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    test_df = pd.read_csv('credit_files/application_test.csv')
    df = pd.read_csv('credit_files/application_train.csv')
    print("Test samples: {}".format(len(test_df)))
    
    # Merging
    df = pd.concat([df,test_df])
    df = df.reset_index()

    # Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    
    #Only keeping relevant columns
    df = df[['EXT_SOURCE_3','EXT_SOURCE_2', 'NAME_EDUCATION_TYPE_Higher education','NAME_INCOME_TYPE_Working',
             'NAME_EDUCATION_TYPE_Secondary / secondary special','CODE_GENDER','NAME_CONTRACT_TYPE_Cash loans',
             'REGION_RATING_CLIENT', 'FLAG_DOCUMENT_3','TARGET']]

    #df = df.dropna(subset=['TARGET','EXT_SOURCE_3','EXT_SOURCE_2','EXT_SOURCE_1'])
    df = df.dropna()

    # Define predictors (feature columns), while exluding payment rate
    predictors = [col for col in df.columns if col not in ['SK_ID_CURR', 'TARGET','PAYMENT_RATE']]

    del test_df
    gc.collect()
    return df

#Checking the first part
trial_1 = application_test()
trial_1

# Split data into features (X) and target (y)
X = trial_1.drop(columns = ['TARGET'])
y = trial_1['TARGET']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the pipeline
pipeline = Pipeline([
    ('scaling', StandardScaler()),   # StandardScaler for numerical features
    ('model', XGBClassifier(
        reg_lambda=2.481367528520412,
        max_depth=14,
        learning_rate=0.2030510614528276,
        n_estimators=173,
        colsample_bytree=0.9173502331327696,
        reg_alpha=0.02759991820225434,
        subsample=0.8330533878126005,
        n_jobs=1
    ))       # XGBoost model
])

pipeline.fit(X_train, y_train)

# Get prediction probabilities
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Apply the optimized threshold (0.15) for final classification
optimal_threshold = 0.15
y_pred = (y_proba >= optimal_threshold).astype(int)

# Define a custom evaluation metric
def custom_metric(y_true, y_pred):
    """
    Score personnalisé pour évaluer le coût métier.

    Règles de scoring :
    - Les vrais positifs (TP) et vrais négatifs (TN) sont récompensés.
    - Les faux positifs (FP) pénalisent (bon client prédit mauvais).
    - Les faux négatifs (FN) pénalisent plus sévèrement (mauvais client prédit bon).

    Ici, nous supposons par exemple que le coût d'un FN est dix fois supérieur à celui d'un FP.

    Args:
        y_true (array-like): Véritables labels (0 ou 1).
        y_pred (array-like): Prédictions (labels binaires).

    Returns:
        score (float): Score calculé.
    """
    # Si y_pred contient des probabilités ou des vecteurs (multi-classe), on transforme en labels
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)  # Classification multi-classes
    else:
        y_pred = np.round(y_pred).astype(int)  # Classification binaire

    # Calcul de la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Poids attribués à chaque issue
    weight_tp = 1    # Récompense pour TP
    weight_tn = 1    # Récompense pour TN
    weight_fp = -1   # Pénalité pour FP
    weight_fn = -10  # Pénalité pour FN (coût dix fois supérieur)

    # Calcul du score personnalisé
    score = (weight_tp * tp) + (weight_tn * tn) + (weight_fp * fp) + (weight_fn * fn)
    return score

# Evaluate the model using the custom metric
custom_score = custom_metric(y_test, y_pred)
print(f"Custom business score after threshold optimization: {custom_score}")

'''
import joblib
joblib.dump(pipeline, '/Users/maudg1/Documents/PythonA_API/pipeline_clients_traintest_4.joblib')


from mlflow.models.signature import infer_signature
signature = infer_signature(X_train, y_train)

mlflow.sklearn.save_model(pipeline, '/Users/maudg1/Documents/PythonA_API/mlflow_model2', signature=signature)

'''