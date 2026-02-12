@echo off
echo Installing Hft2 Frontend Dependencies...
echo.
cd /d "%~dp0trading-dashboard"

echo Installing styled-components...
call npm install styled-components

echo Installing react-hot-toast...
call npm install react-hot-toast

echo Installing chart.js and react-chartjs-2...
call npm install chart.js react-chartjs-2

echo Installing plotly.js and react-plotly.js...
call npm install plotly.js react-plotly.js

echo Installing @types/node for TypeScript...
call npm install --save-dev @types/node

echo.
echo ===================================
echo Installation Complete!
echo ===================================
echo.
pause
