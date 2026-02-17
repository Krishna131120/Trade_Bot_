@echo off
REM Start HFT2 Backend (web_backend.py on port 5000)
REM This starts the backend that handles /api/* endpoints for the BOT page

echo ========================================
echo Starting HFT2 Backend on Port 5000
echo ========================================
echo.

cd backend\hft2\backend

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if port 5000 is already in use
netstat -ano | findstr ":5000" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo WARNING: Port 5000 is already in use!
    echo Please stop the existing process first.
    echo.
    echo To find and kill the process:
    echo   netstat -ano ^| findstr ":5000"
    echo   taskkill /PID ^<PID^> /F
    echo.
    pause
    exit /b 1
)

echo Starting web_backend.py on port 5000...
echo.
echo Backend will be available at:
echo   - API: http://127.0.0.1:5000/api/*
echo   - Docs: http://127.0.0.1:5000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python web_backend.py --host 0.0.0.0 --port 5000

pause
