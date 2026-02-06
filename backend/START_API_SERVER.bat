@echo off
echo Starting Stock Analysis API Server...
echo.
echo This will start the HTTP API server on port 5000
echo Frontend will be able to connect to: http://127.0.0.1:5000
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
python api_wrapper.py

pause