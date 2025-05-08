@echo off
setlocal EnableDelayedExpansion

REM --- Start Timer ---
for /f %%i in ('powershell -command "Get-Date -Format o"') do set START=%%i

echo [1/4] Preprocessing data...
python test_preprocessing.py

echo [2/4] Training model...
python test_train_model.py

echo [3/4] Evaluating model...
python test_evaluate_model.py

echo [4/4] Starting MLflow UI at http://127.0.0.1:5000 ...
call .\.venv\Scripts\activate.bat
start cmd /k "mlflow ui"

REM --- End Timer ---
for /f %%i in ('powershell -command "Get-Date -Format o"') do set END=%%i

echo.
echo ✅ Pipeline complete!
echo Start time: %START%
echo End time:   %END%

pause
