@echo off
rem Laptop 2: serve the UI on http://localhost:5500
rem Usage: deployment\run_ui.bat 192.168.1.20
cd /d "%~dp0\.."
set API_IP=%1
if "%API_IP%"=="" set API_IP=127.0.0.1
echo When the server starts, open: http://localhost:5500/index.html?api=http://%API_IP%:8000
python -m http.server 5500 --directory ui
