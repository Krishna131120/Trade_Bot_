@echo off
REM Run npm command with fixed PATH (temporary fix for current session)
cd /d "%~dp0"

REM Remove problematic PATH entry for this session
set "PATH=%PATH:C:\Program Files\nodejs\node_modules\npm\bin;=%"

REM Use npm.cmd directly with full path
if "%1"=="" (
    echo Usage: run_npm_fixed.bat [npm command]
    echo Example: run_npm_fixed.bat install
    echo Example: run_npm_fixed.bat run dev
    exit /b 1
)

"C:\Program Files\nodejs\npm.cmd" %*
