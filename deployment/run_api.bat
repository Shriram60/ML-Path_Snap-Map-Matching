@echo off
rem Laptop 1: start the PathSnap API on all network interfaces, port 8000.
rem Set PATHSNAP_DATA (and PATHSNAP_EVAL_DIR on demo day) before running.
cd /d "%~dp0\.."
if "%PATHSNAP_DATA%"=="" if "%PATHSNAP_EVAL_DIR%"=="" (
  echo Set PATHSNAP_DATA or PATHSNAP_EVAL_DIR first.
  exit /b 1
)
echo This laptop's IPv4 addresses, give one to Laptop 2:
ipconfig | findstr /i "IPv4"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
