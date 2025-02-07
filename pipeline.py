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
    df = df[['EXT_SOURCE_3','EXT_SOURCE_2', 'NAME_EDUCATION_TYPE_Higher education','CODE_GENDER',
             'NAME_EDUCATION_TYPE_Secondary / secondary special','FLAG_DOCUMENT_3','AMT_REQ_CREDIT_BUREAU_HOUR',
             'REGION_RATING_CLIENT', 'EXT_SOURCE_1', 'NAME_INCOME_TYPE_Working','FLAG_EMP_PHONE','TARGET']]

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
        reg_lambda=2.9254886430096265,
        max_depth=5,
        learning_rate=0.20333096380873394,
        n_estimators=165,
        colsample_bytree=0.8856368296634629,
        reg_alpha=0.02944488920695857,
        subsample=0.9702273343033714,
        n_jobs=1
    ))       # XGBoost model
])

pipeline.fit(X_train, y_train)

pipeline.score(X_test, y_test)

'''
import joblib
joblib.dump(pipeline, '/home/pipeline_clients_traintest_2.joblib')

from mlflow.models.signature import infer_signature
signature = infer_signature(X_train, y_train)

mlflow.sklearn.save_model(pipeline, '/home/mlflow_model', signature=signature)
'''