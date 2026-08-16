@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo Running static code checks on src\ folder
echo ===================================================

REM Ensure we run from repository root
cd /d %~dp0

REM Load environment variables from .env file
for /F "usebackq tokens=1* delims==" %%A in (".env") do (
    REM Skip comment lines (starting with #) or empty lines
    echo %%A | findstr /B "#" >nul || (
        set "%%A=%%~B"
    )
)

REM Set folders
set SRC_DIR=src
set TEST_DIR=src\unit_tests
set REPORT_DIR=reports
REM Ensure Python can import from src/
set PYTHONPATH=%CD%\src

REM Create reports folder
if not exist "%REPORT_DIR%" mkdir "%REPORT_DIR%"

echo.
echo ===== Ruff (style, lint, quality) =====
ruff check "%SRC_DIR%" --fix
if %ERRORLEVEL% NEQ 0 (
    echo Ruff found issues.
) else (
    echo Ruff passed with no issues.
)

echo.
echo ===== pydocstyle (docstrings) =====
pydocstyle "%SRC_DIR%" "%TEST_DIR%" > "%REPORT_DIR%\pydocstyle.txt"
if %ERRORLEVEL% NEQ 0 (
    echo pydocstyle found docstring issues. See "%REPORT_DIR%\pydocstyle.txt".
) else (
    echo pydocstyle passed with no issues.
)

echo.
echo ===== Bandit (security) =====
bandit -r "%SRC_DIR%" -f txt -o "%REPORT_DIR%\bandit.txt"
if %ERRORLEVEL% NEQ 0 (
    echo Bandit found potential security issues.
) else (
    echo Bandit passed with no issues.
)

echo.
echo ===== Pytest =====
coverage erase
pytest "%TEST_DIR%" --maxfail=1 --disable-warnings -q --junitxml="%REPORT_DIR%\junit.xml"
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] Pytest reported failures.
)

echo.
echo Running test coverage...
coverage erase
coverage run -m pytest "%TEST_DIR%"
coverage report -m
coverage xml -o coverage.xml

echo.
echo Generating HTML coverage report...
coverage html
start htmlcov\index.html

echo Reports generated:
echo - coverage.xml (for Sonar)
echo - %REPORT_DIR%\junit.xml (for Sonar)
echo - htmlcov\index.html (human-readable)
echo - %REPORT_DIR%\bandit.txt (security review)

echo All checks complete.
endlocal
