@echo off
setlocal
cd /d "%~dp0.."
set "SRC_DIR=src"
set "REPORT_DIR=outputs\quality"
set "CHECK_FAILED=0"
if not exist "%REPORT_DIR%" mkdir "%REPORT_DIR%"

python -m radon cc "%SRC_DIR%" -e "**/__init__.py" -s -a > "%REPORT_DIR%\cyclomatic_complexity.txt"
if errorlevel 1 set "CHECK_FAILED=1"
python -m radon mi "%SRC_DIR%" -e "**/__init__.py" -s > "%REPORT_DIR%\maintainability_index.txt"
if errorlevel 1 set "CHECK_FAILED=1"
python -m radon raw "%SRC_DIR%" -e "**/__init__.py" > "%REPORT_DIR%\raw_metrics.txt"
if errorlevel 1 set "CHECK_FAILED=1"

echo Reports: %CD%\%REPORT_DIR%
exit /b %CHECK_FAILED%
