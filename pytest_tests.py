import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from pipeline import one_hot_encoder, application_test

def test_one_hot_encoder():
    # Sample dataframe
    data = {
        'A': ['cat', 'dog', np.nan, 'dog'],
        'B': [1, 2, 3, 4],
        'C': ['yes', 'no', 'yes', 'no']
    }
    df = pd.DataFrame(data)

    # Run the one_hot_encoder function
    df_encoded, new_columns = one_hot_encoder(df, nan_as_category=True)

    # Assertions
    assert 'A_cat' in df_encoded.columns
    assert 'A_dog' in df_encoded.columns
    assert 'A_nan' in df_encoded.columns, "NaN category should be included when nan_as_category is True."
    assert 'C_yes' in df_encoded.columns
    assert 'C_no' in df_encoded.columns
    assert 'C_nan' in df_encoded.columns, "NaN category should be included for column C when nan_as_category is True."

    # Check that new columns match the returned new_columns
    expected_new_columns = ['A_cat', 'A_dog', 'A_nan', 'C_yes', 'C_no', 'C_nan']
    assert set(new_columns) == set(expected_new_columns), "Returned new columns do not match expected columns."

    # Check that original columns are not removed
    assert 'B' in df_encoded.columns
    assert len(df_encoded) == len(df), "Row count should remain unchanged."

@pytest.fixture
def mock_csv_data():
    application_train = pd.DataFrame({
        'CODE_GENDER': ['M', 'F', 'XNA'],
        'FLAG_OWN_CAR': ['Y', 'N', 'N'],
        'FLAG_OWN_REALTY': ['Y', 'N', 'Y'],
        'DAYS_EMPLOYED': [365243, -100, -200],
        'EXT_SOURCE_3': [0.1, 0.2, np.nan],
        'EXT_SOURCE_2': [0.3, 0.4, 0.5],
        'NAME_EDUCATION_TYPE': ['Higher education', 'Secondary / secondary special', 'Higher education'],
        'FLAG_DOCUMENT_3': [1, 0, 1],
        'AMT_REQ_CREDIT_BUREAU_HOUR': [0.0, 1.0, np.nan],
        'REGION_RATING_CLIENT': [1, 2, 3],
        'EXT_SOURCE_1': [np.nan, 0.7, 0.6],
        'NAME_INCOME_TYPE': ['Working', 'Working', 'Working'],
        'FLAG_EMP_PHONE': [1, 0, 1],
        'TARGET': [0, 1, 1]
    })

    mock_test_data = pd.DataFrame({
        'CODE_GENDER': ['M', 'F'],
        'FLAG_OWN_CAR': ['N', 'N'],
        'FLAG_OWN_REALTY': ['Y', 'N'],
        'DAYS_EMPLOYED': [-100, -200],
        'EXT_SOURCE_3': [0.3, 0.4],
        'EXT_SOURCE_2': [0.5, 0.6],
        'NAME_EDUCATION_TYPE': ['Higher education', 'Secondary / secondary special'],
        'FLAG_DOCUMENT_3': [1, 1],
        'AMT_REQ_CREDIT_BUREAU_HOUR': [1.0, 0.0],
        'REGION_RATING_CLIENT': [3, 2],
        'EXT_SOURCE_1': [0.8, 0.9],
        'NAME_INCOME_TYPE': ['Working', 'Working'],
        'FLAG_EMP_PHONE': [1, 0]
    })

    return application_train, mock_test_data


def test_application_test(mock_csv_data):
    application_train, mock_test_data = mock_csv_data

    # Mock pd.read_csv
    with patch('pandas.read_csv', side_effect=[mock_test_data, application_train]):
        df = application_test()  # Call the actual function being tested

    # Ensure test data is merged and cleaned properly
    assert 'CODE_GENDER' in df.columns, "CODE_GENDER column should exist."
    assert 'EXT_SOURCE_3' in df.columns, "EXT_SOURCE_3 column should exist."
    assert df['CODE_GENDER'].isnull().sum() == 0, "There should be no missing values in CODE_GENDER after factorization."
    assert 'DAYS_EMPLOYED' not in df.columns, "'DAYS_EMPLOYED' should not be in the final DataFrame."

    # Ensure dropped columns are not present
    assert 'SK_ID_CURR' not in df.columns, "SK_ID_CURR should not be in the resulting dataframe."

    # Ensure only relevant columns remain
    expected_columns = ['EXT_SOURCE_3', 'EXT_SOURCE_2', 'NAME_EDUCATION_TYPE_Higher education',
                        'CODE_GENDER', 'NAME_EDUCATION_TYPE_Secondary / secondary special',
                        'FLAG_DOCUMENT_3', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'REGION_RATING_CLIENT',
                        'EXT_SOURCE_1', 'NAME_INCOME_TYPE_Working', 'FLAG_EMP_PHONE', 'TARGET']
    assert set(expected_columns).issubset(df.columns), "Some expected columns are missing in the final dataframe."