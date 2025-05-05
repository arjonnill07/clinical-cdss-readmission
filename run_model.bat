@echo off
call .\mlops\Scripts\activate.bat
python src\models\train_model.py
