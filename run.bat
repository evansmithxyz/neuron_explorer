@echo off
rem Starts the neuron explorer, then open http://127.0.0.1:8765 in a browser.
cd /d "%~dp0"
start "" http://127.0.0.1:8765
".venv\Scripts\python.exe" server.py
