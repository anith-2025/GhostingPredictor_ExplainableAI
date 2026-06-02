@echo off
REM Packaged installer for Windows (github_release)
cd ..\backend
python -m venv ghost_env
call ghost_env\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r ..\..\requirements.txt
echo Requirements installed.
