@echo off
REM Start api_server.py (trading agent) on port 8000
cd /d "%~dp0"
if exist backend\.venv\Scripts\activate.bat (
  call backend\.venv\Scripts\activate.bat
) else (
  echo No backend\.venv found. Create it and install: pip install -r backend\requirements-unified.txt
  pause
  exit /b 1
)
cd backend
echo Starting api_server.py (trading agent) on http://127.0.0.1:8000 ...
echo Open http://127.0.0.1:8000/docs to verify. Ctrl+C to stop.
python api_server.py
