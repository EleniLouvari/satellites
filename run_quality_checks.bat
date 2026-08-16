@echo off
setlocal

echo Running radon checks for ESA quality metrics...

REM Go to the directory where this .bat file lives (assumed repo root)
cd /d "%~dp0"

REM Define report directory
set "REPORT_DIR=src\notebooks\quality_reports"

REM Create the directory (and parents if needed)
if not exist "%REPORT_DIR%" (
    mkdir "%REPORT_DIR%"
)

echo Reports will be saved under: %CD%\%REPORT_DIR%
echo.

echo Analyzing Cyclomatic Complexity...
radon cc src -e "**/tests/**,**/unit_tests/**,**/notebooks/**,**/__init__.py" -s -a ^
  > "%REPORT_DIR%\cyclomatic_complexity.txt"

echo Analyzing Maintainability Index...
radon mi src -e "**/tests/**,**/unit_tests/**,**/notebooks/**,**/__init__.py" -s ^
  > "%REPORT_DIR%\maintainability_index.txt"

echo Analyzing Raw Metrics...
radon raw src -e "**/tests/**,**/unit_tests/**,**/notebooks/**,**/__init__.py" ^
  > "%REPORT_DIR%\raw_metrics.txt"

echo.
echo All reports saved in: %CD%\%REPORT_DIR%

endlocal
