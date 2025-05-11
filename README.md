# MLOps Heart Disease Project

This repository contains an MLOps project for predicting heart disease based on patient data.

Dataset source: [Kaggle Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

Struktura projektu:

MLOps_HearDisease/
├── .github
│   └── workflows
│       └── CI_CD_pipeline.yml
├── data/                # dane surowe i przetworzone
│   ├── raw/
│   └── processed/
├── models/
│
│
├── src/                 # cały kod (preprocessing, trening modeli, itd.)
│   ├── data/
│   ├──models/
        ├── train_model_LR.py
        └── train_model_RFC.py
│   ├── features/
│   └── pipelines/
│
├── notebooks/           # notebooki do eksploracji danych
│
├── mlruns/              # katalog MLflow (powstanie automatycznie później)
│
├── README.md            # dokumentacja projektu
├── requirements.txt     # lista bibliotek
├── .gitignore           # plik ignorujący niepotrzebne rzeczy           
├── evaluate_model.py
├── run_pipeline.bat
├── test_evaluate_model.py
├── test_load_data.py
├── test_preprocessing.py
├── test_split_model.py
└── test_train_model.py  
