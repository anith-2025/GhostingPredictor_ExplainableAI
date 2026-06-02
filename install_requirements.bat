@echo off
REM Quick installer for Windows (run from project root)
python -m venv ghost_env
call ghost_env\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Requirements installed. Activate with:
echo    call ghost_env\Scripts\activate.bat
pause
