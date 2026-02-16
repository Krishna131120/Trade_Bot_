@echo off
REM Fix npm prefix configuration
echo Fixing npm prefix configuration...
echo.

REM Set npm prefix to Node.js installation directory
"C:\Program Files\nodejs\npm.cmd" config set prefix "C:\Program Files\nodejs"

echo.
echo npm prefix set to: C:\Program Files\nodejs
echo.
echo Verifying npm works...
"C:\Program Files\nodejs\npm.cmd" --version

echo.
echo If npm still doesn't work, try:
echo   1. Reinstall Node.js (download from nodejs.org)
echo   2. During installation, ensure "Add to PATH" is checked
echo   3. Restart your computer after installation
echo.
pause
