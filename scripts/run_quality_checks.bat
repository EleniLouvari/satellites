@echo off
setlocal
REM Optional first argument: Python executable (the notebook passes sys.executable).
set "CHECK_PYTHON=python"
if not "%~1"=="" set "CHECK_PYTHON=%~1"
cd /d "%~dp0.." || exit /b 2
set "SRC_DIR=src"
set "REPORT_DIR=outputs\quality"
set "CHECK_FAILED=0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
if not exist "%REPORT_DIR%" mkdir "%REPORT_DIR%"
if not exist "%REPORT_DIR%" exit /b 2

"%CHECK_PYTHON%" -c "import importlib.util, sys; available = importlib.util.find_spec('radon') is not None; print('Python: ' + sys.executable if available else 'Error: Missing radon. Install the development dependencies with python -m pip install -e .[dev].'); sys.exit(0 if available else 2)"
if errorlevel 1 exit /b 2

echo Measuring cyclomatic complexity...
"%CHECK_PYTHON%" -m radon cc "%SRC_DIR%" -e "**/__init__.py" -s -a > "%REPORT_DIR%\cyclomatic_complexity.txt" 2> "%REPORT_DIR%\cyclomatic_complexity.stderr.txt"
if errorlevel 1 set "CHECK_FAILED=1"
echo Measuring maintainability...
"%CHECK_PYTHON%" -m radon mi "%SRC_DIR%" -e "**/__init__.py" -s > "%REPORT_DIR%\maintainability_index.txt" 2> "%REPORT_DIR%\maintainability_index.stderr.txt"
if errorlevel 1 set "CHECK_FAILED=1"
echo Measuring raw metrics...
"%CHECK_PYTHON%" -m radon raw "%SRC_DIR%" -e "**/__init__.py" > "%REPORT_DIR%\raw_metrics.txt" 2> "%REPORT_DIR%\raw_metrics.stderr.txt"
if errorlevel 1 set "CHECK_FAILED=1"

echo Reports: %CD%\%REPORT_DIR%
if "%CHECK_FAILED%"=="0" (echo Quality reports generated.) else (echo Error: A Radon check failed. See the stderr reports.)
exit /b %CHECK_FAILED%
