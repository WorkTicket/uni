@echo off
rem No build needed: first run installs dependencies into a local venv, later runs start instantly. Requires Python 3.10+ on PATH.
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  python -m venv .venv || goto :err
  .venv\Scripts\python -m pip install -e . uvicorn fastapi websockets httpx || goto :err
)
.venv\Scripts\python -m universe_engine.launcher
exit /b 0
:err
echo Setup failed. Is Python 3.10+ installed and on PATH?
pause
exit /b 1
