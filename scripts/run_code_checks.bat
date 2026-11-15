@echo off
setlocal enabledelayedexpansion
REM Optional arguments, in any order:
REM   <python.exe>  Python executable to use (default: python)
REM   live | file   pytest output in the terminal (default) or in outputs\quality\pytest.txt
REM   db            also run the parcel_app PostgreSQL integration tests against a throwaway
REM                 Docker container (postgres:16-alpine on 127.0.0.1:%PARCEL_TEST_DB_PORT%, default 55439)
REM Capture the script folder before "shift", which also shifts %0.
set "SCRIPT_DIR=%~dp0"
set "CHECK_PYTHON=python"
set "PYTEST_OUTPUT_MODE=live"
set "DB_TESTS=0"
set "PYTHON_ARG_SET=0"

:parse_args
if "%~1"=="" goto :args_done
if /I "%~1"=="live" (set "PYTEST_OUTPUT_MODE=live" & goto :next_arg)
if /I "%~1"=="file" (set "PYTEST_OUTPUT_MODE=file" & goto :next_arg)
if /I "%~1"=="db" (set "DB_TESTS=1" & goto :next_arg)
if not "%PYTHON_ARG_SET%"=="0" goto :invalid_arg
REM Set outside a parenthesized block so paths such as "Program Files (x86)" stay intact.
set "CHECK_PYTHON=%~1"
set "PYTHON_ARG_SET=1"
:next_arg
shift
goto :parse_args
:invalid_arg
echo Error: Invalid argument "%~1". Use a Python executable, "live", "file" or "db".
exit /b 2
:args_done

REM Resolve repository root relative to this script location
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

if "%DB_TESTS%"=="1" (
    echo.
    echo ===== Throwaway test database ^(Docker^) =====
    call :start_test_db
)
if "%DB_TESTS%"=="1" if errorlevel 1 (
    call :stop_test_db
    echo Error: Database integration tests could not start; no tests were run.
    exit /b 2
)
if "%DB_TESTS%"=="0" echo.
if "%DB_TESTS%"=="0" echo Database integration tests are skipped; add the "db" argument to run them.

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
if "%DB_TESTS%"=="1" call :stop_test_db
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


REM ---------------------------------------------------------------------------
REM Throwaway PostgreSQL for the parcel_app integration tests. The tests create
REM and delete rows, so they must never reach the database configured in .env:
REM every database setting is overridden to point at this container only.
REM ---------------------------------------------------------------------------
:start_test_db
if not defined PARCEL_TEST_DB_PORT set "PARCEL_TEST_DB_PORT=55439"
set "TEST_DB_CONTAINER=satellites-code-checks-db"
docker version >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not available. Start Docker Desktop, or run without "db".
    exit /b 1
)
REM Remove a container left behind by an interrupted earlier run.
docker rm -f %TEST_DB_CONTAINER% >nul 2>&1
docker run -d --rm --name %TEST_DB_CONTAINER% -p 127.0.0.1:%PARCEL_TEST_DB_PORT%:5432 -e POSTGRES_PASSWORD=code-checks postgres:16-alpine >nul
if errorlevel 1 (
    echo Error: Could not start the test database container on port %PARCEL_TEST_DB_PORT%. Set PARCEL_TEST_DB_PORT to a free port.
    exit /b 1
)
REM The image initializes first; TCP connections are accepted only by the final server.
set /a "TEST_DB_WAIT=0"
:wait_test_db
docker exec %TEST_DB_CONTAINER% pg_isready -U postgres -h 127.0.0.1 >nul 2>&1
if not errorlevel 1 goto :test_db_ready
set /a "TEST_DB_WAIT+=1"
if %TEST_DB_WAIT% GEQ 60 (
    echo Error: The test database did not become ready within 60 seconds.
    exit /b 1
)
REM ping is used as a delay because timeout fails when input is redirected.
ping -n 2 127.0.0.1 >nul
goto :wait_test_db
:test_db_ready
set "PARCEL_TEST_DB=1"
set "POSTGRES_HOST=127.0.0.1"
set "POSTGRES_PORT=%PARCEL_TEST_DB_PORT%"
set "POSTGRES_USER=postgres"
set "POSTGRES_PASSWORD=code-checks"
set "POSTGRES_DB=postgres"
set "DB_NAME=postgres"
"%CHECK_PYTHON%" -m parcel_app.manage migrate
if errorlevel 1 (
    echo Error: Applying the parcel_app migrations to the test database failed.
    exit /b 1
)
echo Test database ready on 127.0.0.1:%PARCEL_TEST_DB_PORT% ^(container %TEST_DB_CONTAINER%^).
exit /b 0

:stop_test_db
REM Started with --rm, so stopping also deletes the container and its data.
docker stop %TEST_DB_CONTAINER% >nul 2>&1
echo Test database container removed.
exit /b 0
