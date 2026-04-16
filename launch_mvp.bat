@echo off
setlocal ENABLEDELAYEDEXPANSION

cd /d "%~dp0"

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  set "PYTHON=python"
)

:menu
cls
echo ============================================
echo   Mouse Vision MVP Launcher
echo ============================================
echo.
echo 1^) Download required data
echo 2^) Run full pipeline
echo 3^) Launch scientist dashboard
echo 4^) Download data + pipeline + dashboard
echo 5^) Create desktop shortcut
echo 6^) Exit
echo.
set /p CHOICE=Choose an option [1-6]: 

echo.
if "%CHOICE%"=="1" goto download
if "%CHOICE%"=="2" goto pipeline
if "%CHOICE%"=="3" goto dashboard
if "%CHOICE%"=="4" goto all
if "%CHOICE%"=="5" goto shortcut
if "%CHOICE%"=="6" goto done

echo Invalid choice.
pause
goto menu

:download
echo Running data download...
"%PYTHON%" scripts\download_data.py --config configs\mvp_config.json
pause
goto menu

:pipeline
echo Running full pipeline...
"%PYTHON%" scripts\run_pipeline.py --config configs\mvp_config.json
pause
goto menu

:dashboard
echo Launching dashboard...
"%PYTHON%" -m streamlit run app\scientist_dashboard.py
goto menu

:all
echo Downloading data...
"%PYTHON%" scripts\download_data.py --config configs\mvp_config.json
echo.
echo Running full pipeline...
"%PYTHON%" scripts\run_pipeline.py --config configs\mvp_config.json
echo.
echo Launching dashboard...
"%PYTHON%" -m streamlit run app\scientist_dashboard.py
goto menu

:shortcut
echo Creating desktop shortcut...
powershell -ExecutionPolicy Bypass -File scripts\create_desktop_shortcut.ps1
pause
goto menu

:done
echo Bye.
endlocal
