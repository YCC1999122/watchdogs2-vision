@echo off
REM Side-by-side demo: original video (left) vs NetHack view (right).
REM Uses project-local Python only; no system environment required.
set ROOT=%~dp0
"%ROOT%runtime\python\python.exe" "%ROOT%run.py" --source "%ROOT%test_assets\raw\street_city_855564.mp4" --compare --display --out "%ROOT%output\compare_street_city.mp4"
pause
