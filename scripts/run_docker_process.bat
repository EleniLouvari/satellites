@echo off
setlocal DisableDelayedExpansion

REM Edit these settings before running. Host paths must be absolute.
set "IMAGE=satellites-parcel-stats:local"
set "INPUT_FOLDER=C:\work_sat\parcel-app\input"
set "OUTPUT_FOLDER=C:\work_sat\parcel-app\runs"
set "CONFIG_FILE=config.json"
set "RUN_NAME=run-001"
set "APP_USER_ID=1"
set "ENV_FILE=%~dp0.env_docker"
set "MEMORY=50g"
set "CPUS=12"
REM Set to 0 when calling from automation instead of double-clicking.
set "PAUSE_AT_END=1"

REM Optional argument: run (default), validate, or resume.
set "ACTION=run"
if not "%~1"=="" set "ACTION=%~1"
if /i "%ACTION%"=="run" goto :check_inputs
if /i "%ACTION%"=="validate" goto :check_inputs
if /i "%ACTION%"=="resume" goto :check_inputs
echo Usage: %~nx0 [run^|validate^|resume]
set "RESULT=2"
goto :done

:check_inputs
set "RESULT=2"
if not exist "%INPUT_FOLDER%\" (
    echo Error: Input folder does not exist: "%INPUT_FOLDER%"
    goto :done
)
if /i "%ACTION%"=="resume" goto :check_accounts
if not exist "%INPUT_FOLDER%\%CONFIG_FILE%" (
    echo Error: Configuration file does not exist: "%INPUT_FOLDER%\%CONFIG_FILE%"
    goto :done
)
if /i "%ACTION%"=="validate" goto :check_docker

:check_accounts
if not exist "%ENV_FILE%" (
    echo Error: Create "%ENV_FILE%" with the database settings and encryption key.
    echo See docs\data_preparation\openeo\DOCKER.md for the file format.
    goto :done
)
if /i "%ACTION%"=="run" goto :check_docker
if not exist "%OUTPUT_FOLDER%\%RUN_NAME%\run.json" (
    echo Error: No saved run found in "%OUTPUT_FOLDER%\%RUN_NAME%".
    goto :done
)

:check_docker
docker info >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is unavailable. Start Docker Desktop with Linux containers.
    goto :done
)
docker image inspect "%IMAGE%" >nul 2>&1
if errorlevel 1 (
    echo Error: Image "%IMAGE%" is not available locally. Build it from the repository root:
    echo docker build -t "%IMAGE%" .
    goto :done
)
if /i "%ACTION%"=="validate" goto :validate
if not exist "%OUTPUT_FOLDER%\" mkdir "%OUTPUT_FOLDER%"
if not exist "%OUTPUT_FOLDER%\" (
    echo Error: Could not create output folder: "%OUTPUT_FOLDER%"
    goto :done
)

set "CLI_ARGS=run --config "/data/input/%CONFIG_FILE%" --output-dir "/data/runs/%RUN_NAME%""
if /i "%ACTION%"=="resume" set "CLI_ARGS=resume --run-dir "/data/runs/%RUN_NAME%""
echo Starting parcel statistics: %ACTION%
REM Results and logs persist in the host output folder after container removal.
docker run --rm --init --stop-timeout 60 --memory="%MEMORY%" --cpus="%CPUS%" ^
    --env-file "%ENV_FILE%" ^
    --mount "type=bind,source=%INPUT_FOLDER%,target=/data/input,readonly" ^
    --mount "type=bind,source=%OUTPUT_FOLDER%,target=/data/runs" ^
    "%IMAGE%" %CLI_ARGS% --app-user-id "%APP_USER_ID%"
set "RESULT=%ERRORLEVEL%"
goto :report

:validate
REM Offline validation needs neither credentials nor a writable output mount.
docker run --rm --init --network none --memory="%MEMORY%" --cpus="%CPUS%" ^
    --mount "type=bind,source=%INPUT_FOLDER%,target=/data/input,readonly" ^
    "%IMAGE%" validate --config "/data/input/%CONFIG_FILE%" --output-dir "/data/runs/%RUN_NAME%"
set "RESULT=%ERRORLEVEL%"

:report
if not "%RESULT%"=="0" (
    echo Error: Container process failed with code %RESULT%.
    goto :done
)
if /i "%ACTION%"=="validate" (
    echo Configuration validated successfully.
) else (
    echo Processing completed successfully. Results: "%OUTPUT_FOLDER%\%RUN_NAME%"
)

:done
if "%PAUSE_AT_END%"=="1" pause
endlocal & exit /b %RESULT%
