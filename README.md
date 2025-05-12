# MLOps Heart Disease Project

## Project Description
This repository contains an MLOps project for predicting heart disease based on patient data. The goal of the project is to build a pipeline that processes data, trains machine learning models, evaluates the models, and logs the results using MLflow. The project also includes the implementation of CI/CD for model deployment and testing.

**Dataset source:** [Kaggle Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

## Project Structure
MLOps_HearDisease/
├── .github/
│   └── workflows/
│       ├── init.yml                  # Initial workflow (e.g. testing, linting)
│       └── train_andevaluate.yml      # Workflow for training and evaluating models
├── data/                               # Raw and processed data
│   ├── raw/                           # Raw data
│   └── processed/                     # Processed data (after preprocessing)
│   └── load_dataset.py                # Script for loading the dataset
├── figures/                            # Folder for storing generated figures (e.g., plots)
├── mlruns/                             # MLflow directory (automatically generated)
├── models/                             # Folder for storing saved models
├── notebooks/                          # Jupyter notebooks for data exploration
├── src/                                # Source code (data processing, model training, etc.)
│   ├── data/                           # Data loading and preprocessing
│   ├── models/                         # Model training scripts
│   │   ├── train_model_LR.py           # Logistic Regression model training
│   │   └── train_model_RFC.py          # Random Forest model training
│   ├── features/                       # Feature engineering
│   ├── pipelines/                      # Code for pipelines
├── .gitignore                          # Files to be ignored by Git
├── Makefile                            # Automation for common tasks
├── README.md                           # Project documentation
├── requirements.txt                    # List of dependencies
├── run_pipeline.bat                    # Batch script to run the pipeline
├── evaluate_model.py                   # Script for model evaluation
├── test_evaluate_model.py              # Test for model evaluation
├── test_load_data.py                   # Test for loading data
├── test_preprocessing.py               # Test for preprocessing data
├── test_split_model.py                 # Test for model splitting
└── test_train_model.py                 # Test for training models

## Requirements
To run the project, you need to install the required libraries. You can do this by running the following command: 
pip install -r requirements.txt

Run Instructions
1. Exploratory Data Analysis (EDA)
Jupyter notebooks for data exploration can be found in the notebooks/ folder.
In these notebooks, you will explore the dataset, visualize the data, and perform feature engineering.

2. Data Preprocessing
The test_preprocessing.py script handles data preprocessing (e.g., cleaning, scaling, and splitting data).
The processed data is saved in the data/processed/ directory for further use in model training.

3. Model Training
The training scripts for the models can be found in the src/models/ folder.
To train a Logistic Regression model, use the train_model_LR.py.
To train a Random Forest Classifier model, use the train_model_RFC.py.
Each training script saves the trained model in the models/ directory and logs metrics to MLflow.

4. Model Evaluation
The evaluate_model.py script allows you to evaluate a trained model using test data.
You can specify the model type (e.g., Logistic Regression, Random Forest) when calling this script.
Evaluation metrics include Accuracy, Precision, Recall, F1 Score, and ROC AUC, and the results are logged in MLflow.

5. CI/CD Pipeline
The CI/CD pipeline is configured in .github/workflows/ folder.
There are two workflows: 
        init.yml: Runs linting, formatting checks, and other tests.
        train_andevaluate.yml: Trains and evaluates models automatically in the CI/CD pipeline.

6. Running the Pipeline
To run the pipeline locally, use the run_pipeline.bat script.
This will trigger the data processing, model training, and evaluation processes automatically.

## MLFlow Tracking
MLflow is used for tracking experiments and logging models. The following can be logged:
Metrics: Accuracy, Precision, Recall, F1 Score, ROC AUC.
Parameters: Model type, hyperparameters.
Artifacts: Trained models.

By default, MLflow logs the metrics and models in the mlruns/ directory.