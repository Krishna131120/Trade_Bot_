@echo off
REM Master startup script: starts all services in order
REM 1. HFT2 backend (web_backend on 5000, HFT2 API on 5001)
REM 2. api_server.py (trading agent on 8000)
REM 3. Frontend (on 5173)

cd /d "%~dp0"

echo ========================================
echo Starting Trade Bot System
echo ========================================
echo.
echo Step 1: Starting HFT2 backend (ports 5000, 5001)...
echo.

REM Start HFT2 backend in a new window
start "HFT2 Backend" cmd /k "cd /d %~dp0backend\hft2\backend && ..\..\.venv\Scripts\activate.bat && python run_hft2.py"

REM Wait for backend to start
timeout /t 10 /nobreak >nul

echo.
echo Step 2: Starting api_server.py (trading agent on port 8000)...
echo.

REM Start api_server in a new window
start "API Server (Trading Agent)" cmd /k "cd /d %~dp0backend && .venv\Scripts\activate.bat && python api_server.py"

REM Wait for api_server to start
timeout /t 5 /nobreak >nul

echo.
echo Step 3: Starting frontend (port 5173)...
echo.

REM Start frontend in a new window (use run_frontend.bat which handles npm path correctly)
start "Frontend" cmd /k "cd /d %~dp0 && call run_frontend.bat"

echo.
echo ========================================
echo All services started!
echo ========================================
echo.
echo Backend (auth/login): http://127.0.0.1:5000/docs
echo API Server (trading): http://127.0.0.1:8000/docs
echo Frontend: http://127.0.0.1:5173
echo.
echo Press any key to exit this window (services will keep running)...
pause >nul
