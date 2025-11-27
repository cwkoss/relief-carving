@echo off
echo ========================================
echo Relief Carving GUI Setup
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8 or later.
    pause
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo Checking GUI dependencies...
python -c "import gradio" >nul 2>&1
if errorlevel 1 (
    echo.
    echo GUI dependencies not installed.
    echo.
    echo Please run this command in a regular Command Prompt:
    echo   python -m pip install gradio opencv-python pillow numpy
    echo.
    echo Then run this script again.
    pause
    exit /b 1
)

echo GUI dependencies OK
echo.
echo ========================================
echo Starting Relief Carving GUI...
echo ========================================
echo.
echo The GUI will open in your browser at:
echo http://127.0.0.1:7860
echo.
echo Press Ctrl+C to stop the GUI
echo ========================================
echo.

python gradio_gui.py

pause
