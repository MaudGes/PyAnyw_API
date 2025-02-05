Predicting Loan Repayment Using Machine Learning

This project focuses on developing a machine learning model to predict whether a client is likely to repay a bank loan. The goal is to use a large dataset containing detailed information about various clients to train and evaluate the model, ultimately selecting the most effective solution for this classification problem.
Key Components of the Project

    Feature Engineering and Initial Model Testing:
        File: Features Eng+New models.ipynb
        This notebook includes the preprocessing and feature engineering steps applied to the dataset. It involves transforming raw data into meaningful features that enhance the performance of machine learning models.
        Additionally, preliminary models are trained and tested to assess their performance and suitability for the task.

    Final Model Selection and Hyperparameter Tuning:
        File: Final_Models.ipynb
        This notebook explores various machine learning models, comparing their performance to identify the most promising candidates.
        A thorough hyperparameter tuning process is conducted to optimize the selected models, ensuring the best possible performance for loan repayment prediction.

    Exported Final Model:
        Folder: mlflow_model
        The final chosen model is XGBoost, a highly effective gradient boosting algorithm.
        The model is exported using MLflow, a platform for managing machine learning workflows. The folder contains the trained model, all associated dependencies, and metadata, making it portable and ready for deployment in production environments.

    Dependency Management:
        File: requirements.txt
        This file lists all the Python packages used in the project along with their specific versions. It ensures that the environment can be reliably reproduced for future work or deployment.

    Prediction API :
        app_p7.py contains the script that allows streamlit to connecte to the API set up in Azure to get a prediction using a json formatted request.

    Dockerfile : 
        To send the model to azure's container.
    

Highlights

    The project employs advanced machine learning techniques and best practices, including feature engineering, hyperparameter tuning, and model management with MLflow.
    The use of XGBoost as the final model reflects its capability to handle complex datasets and deliver robust performance in classification tasks.
    With a clearly defined workflow and organized project structure, this solution is well-suited for scaling and adapting to real-world banking applications.

This project provides a comprehensive approach to predicting loan repayment, paving the way for data-driven decision-making in financial institutions. The API can be used to get a prediction on whether someone will be able to repay a loan or not.

Thank you !
