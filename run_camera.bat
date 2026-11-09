@echo off
REM Real-time camera demo (DirectShow). Simulates head-mounted device input.
REM Uses project-local Python only; no system environment required.
set ROOT=%~dp0
"%ROOT%runtime\python\python.exe" "%ROOT%run.py" --source 0 --compare --display
pause
