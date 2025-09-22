echo Starting D4Xgui...
@echo off
REM D4Xgui Application Launcher for Windows
REM Double-click this file to run the application

if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate
pip install -r requirements.txt
python run.py
pause