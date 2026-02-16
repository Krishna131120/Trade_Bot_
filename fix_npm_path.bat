@echo off
REM Fix npm PATH issue permanently
echo Fixing npm PATH issue...
echo.

REM Get current user PATH
for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USER_PATH=%%B"

REM Remove the problematic npm bin entry
set "USER_PATH=%USER_PATH:C:\Program Files\nodejs\node_modules\npm\bin;=%"
set "USER_PATH=%USER_PATH:;C:\Program Files\nodejs\node_modules\npm\bin=%"

REM Ensure correct Node.js path is present
echo %USER_PATH% | findstr /C:"C:\Program Files\nodejs" >nul
if errorlevel 1 (
    set "USER_PATH=C:\Program Files\nodejs;%USER_PATH%"
)

REM Update user PATH
reg add "HKCU\Environment" /v Path /t REG_EXPAND_SZ /d "%USER_PATH%" /f >nul

echo.
echo PATH fixed! Please restart your terminal/PowerShell for changes to take effect.
echo.
echo After restarting, run: npm --version
pause
