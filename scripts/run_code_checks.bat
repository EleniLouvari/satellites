@echo off
setlocal enabledelayedexpansion
REM Optional first argument: Python executable.
set "CHECK_PYTHON=python"
set "PYTEST_OUTPUT_MODE=live"
set "ARG1=%~1"
set "ARG2=%~2"

if /I "%ARG1%"=="live" (
    set "PYTEST_OUTPUT_MODE=live"
) else if /I "%ARG1%"=="file" (
    set "PYTEST_OUTPUT_MODE=file"
) else if not "%ARG1%"=="" (
    set "CHECK_PYTHON=%ARG1%"
)

if not "%ARG2%"=="" (
    if /I "%ARG2%"=="live" (
        set "PYTEST_OUTPUT_MODE=live"
    ) else if /I "%ARG2%"=="file" (
        set "PYTEST_OUTPUT_MODE=file"
    ) else (
        echo Error: Invalid pytest output mode "%ARG2%". Use "live" or "file".
        exit /b 2
    )
)

REM Resolve repository root relative to this script location
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
cd /d "%REPO_ROOT%" || exit /b 2

echo ===================================================
echo Running static code checks on src\ folder
echo ===================================================

REM Load environment variables from .env file
if exist "%REPO_ROOT%\.env" (
    for /F "usebackq tokens=1* delims==" %%A in ("%REPO_ROOT%\.env") do (
        REM Skip comment lines (starting with #) or empty lines
        echo %%A | findstr /B "#" >nul || (
            set "%%A=%%~B"
        )
    )
)

REM Set folders
set "SRC_DIR=%CD%\src"
set "TEST_DIR=%CD%\tests"
set "REPORT_DIR=%CD%\outputs\quality"

REM Ensure Python can import from src/
set "PYTHONPATH=%CD%\src"

REM Create reports folder
if not exist "%REPORT_DIR%" mkdir "%REPORT_DIR%"
if not exist "%REPORT_DIR%" exit /b 2

"%CHECK_PYTHON%" -c "import sys; print('Python: ' + sys.executable)"
if %ERRORLEVEL% NEQ 0 (
    echo Error: Could not run the configured Python executable "%CHECK_PYTHON%".
    exit /b 2
)

set "CHECK_FAILED=0"

echo.
echo ===== Ruff (style, lint, quality) =====
"%CHECK_PYTHON%" -m ruff check "%SRC_DIR%" --fix > "%REPORT_DIR%\ruff.txt" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Ruff found issues.
    set "CHECK_FAILED=1"
) else (
    echo Ruff passed with no issues.
)

echo.
echo ===== pydocstyle (docstrings) =====
"%CHECK_PYTHON%" -m pydocstyle "%SRC_DIR%" "%TEST_DIR%" > "%REPORT_DIR%\pydocstyle.txt" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo pydocstyle found docstring issues. See "%REPORT_DIR%\pydocstyle.txt".
    set "CHECK_FAILED=1"
) else (
    echo pydocstyle passed with no issues.
)

echo.
echo ===== Bandit (security) =====
"%CHECK_PYTHON%" -m bandit -r "%SRC_DIR%" -f txt -o "%REPORT_DIR%\bandit.txt" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Bandit found potential security issues.
    set "CHECK_FAILED=1"
) else (
    echo Bandit passed with no issues.
)

echo.
echo ===== Pytest with Coverage =====
set "SATELLITES_DISABLE_HTML_REPORT_OPEN=1"
"%CHECK_PYTHON%" -m coverage erase
if /I "%PYTEST_OUTPUT_MODE%"=="file" (
    echo Running pytest with output redirected to "%REPORT_DIR%\pytest.txt"...
    "%CHECK_PYTHON%" -m coverage run --rcfile="%REPO_ROOT%\pyproject.toml" -m pytest "%TEST_DIR%" --maxfail=1 --junitxml="%REPORT_DIR%\junit.xml" > "%REPORT_DIR%\pytest.txt" 2>&1
) else (
    echo Running pytest with live terminal output...
    "%CHECK_PYTHON%" -m coverage run --rcfile="%REPO_ROOT%\pyproject.toml" -m pytest "%TEST_DIR%" --maxfail=1 --junitxml="%REPORT_DIR%\junit.xml"
)
set "PYTEST_EXIT_CODE=%ERRORLEVEL%"
if not "%PYTEST_EXIT_CODE%"=="0" (
    echo [WARN] Pytest reported failures.
)

"%CHECK_PYTHON%" -m coverage report --rcfile="%REPO_ROOT%\pyproject.toml" -m > "%REPORT_DIR%\coverage.txt" 2>&1
if %ERRORLEVEL% NEQ 0 set "CHECK_FAILED=1"
"%CHECK_PYTHON%" -m coverage xml --rcfile="%REPO_ROOT%\pyproject.toml" -o "%REPORT_DIR%\coverage.xml" >nul 2>&1
if %ERRORLEVEL% NEQ 0 set "CHECK_FAILED=1"
set "SATELLITES_DISABLE_HTML_REPORT_OPEN="

echo.
echo Generating HTML coverage report...
"%CHECK_PYTHON%" -m coverage html --rcfile="%REPO_ROOT%\pyproject.toml" -d "%REPORT_DIR%\htmlcov" >nul 2>&1
if %ERRORLEVEL% NEQ 0 set "CHECK_FAILED=1"
echo Coverage HTML generated at: "%REPORT_DIR%\htmlcov\index.html"

echo Reports generated:
echo - %REPORT_DIR%\ruff.txt (style, lint, quality)
echo - %REPORT_DIR%\pydocstyle.txt (docstrings)
if /I "%PYTEST_OUTPUT_MODE%"=="file" (
    echo - %REPORT_DIR%\pytest.txt - pytest results
) else (
    echo - Pytest output is shown live in terminal
)
echo - %REPORT_DIR%\bandit.txt (security review)
echo - %REPORT_DIR%\coverage.xml (for Sonar)
echo - %REPORT_DIR%\junit.xml (for Sonar)
echo - %REPORT_DIR%\htmlcov\index.html (human-readable)

if not "%PYTEST_EXIT_CODE%"=="0" (
    set "CHECK_FAILED=%PYTEST_EXIT_CODE%"
)

if "%CHECK_FAILED%"=="0" (
    echo All checks complete.
) else (
    echo Error: Code checks completed with failures.
)
endlocal & exit /b %CHECK_FAILED%
