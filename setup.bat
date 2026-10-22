@echo off
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-cache-dir -e .
echo Виртуальное окружение создано. Зависимости установлены.
