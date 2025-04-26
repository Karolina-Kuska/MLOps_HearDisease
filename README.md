# MLOps Heart Disease Project

This repository contains an MLOps project for predicting heart disease based on patient data.

Dataset source: [Kaggle Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

Struktura projektu:

MLOps_HearDisease/
│
├── data/                # dane surowe i przetworzone
│   ├── raw/
│   └── processed/
│
├── src/                 # cały kod (preprocessing, trening modeli, itd.)
│   ├── data/
│   ├── models/
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
└── setup.py             # opcjonalnie, jak chcesz mieć instalowalny projekt
