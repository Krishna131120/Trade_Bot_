@echo off
REM Run ONLY the web backend on port 5000 (auth + /docs). Use this to see errors if run_hft2 doesn't bring up 5000.
cd /d "%~dp0"
if exist backend\.venv\Scripts\activate.bat (
  call backend\.venv\Scripts\activate.bat
) else (
  echo No backend\.venv found. Create it and install: pip install -r backend\requirements-unified.txt
  pause
  exit /b 1
)
cd backend\hft2\backend
echo Starting web backend on http://127.0.0.1:5000 ...
echo Open http://127.0.0.1:5000/docs to verify. Ctrl+C to stop.
python web_backend.py --host 0.0.0.0 --port 5000
