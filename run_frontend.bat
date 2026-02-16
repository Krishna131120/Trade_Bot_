@echo off
REM Run frontend dev server - fixes npm PATH issue
cd /d "%~dp0\trading-dashboard"

REM Fix PATH for this session (remove incorrect npm bin entry)
set "PATH=%PATH:C:\Program Files\nodejs\node_modules\npm\bin;=%"
set "PATH=%PATH:;C:\Program Files\nodejs\node_modules\npm\bin=%"

REM Ensure Node.js is in PATH
echo %PATH% | findstr /C:"C:\Program Files\nodejs" >nul
if errorlevel 1 (
    set "PATH=C:\Program Files\nodejs;%PATH%"
)

REM Run npm with full path to avoid PATH issues
"C:\Program Files\nodejs\npm.cmd" run dev
