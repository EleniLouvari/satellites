@echo off
setlocal
cd /d "%~dp0.."
set "SRC_DIR=src\satellites"
set "TEST_DIR=tests"
set "REPORT_DIR=outputs\quality"
set "PYTHONPATH=%CD%\src;%CD%\src\_compat;%PYTHONPATH%"
set "CHECK_FAILED=0"
if not exist "%REPORT_DIR%" mkdir "%REPORT_DIR%"

echo Running checks using the active Python environment...
python -m ruff check "%SRC_DIR%" "%TEST_DIR%" > "%REPORT_DIR%\ruff.txt" 2>&1
if errorlevel 1 set "CHECK_FAILED=1"
python -m pydocstyle "%SRC_DIR%" > "%REPORT_DIR%\pydocstyle.txt" 2>&1
if errorlevel 1 set "CHECK_FAILED=1"
python -m bandit -r "%SRC_DIR%" -f txt -o "%REPORT_DIR%\bandit.txt"
if errorlevel 1 set "CHECK_FAILED=1"
python -m pytest "%TEST_DIR%" --cov=satellites --cov-report=term-missing --cov-report=xml:"%REPORT_DIR%\coverage.xml" --cov-report=html:"%REPORT_DIR%\htmlcov" --junitxml="%REPORT_DIR%\junit.xml"
if errorlevel 1 set "CHECK_FAILED=1"

echo Reports: %CD%\%REPORT_DIR%
exit /b %CHECK_FAILED%
