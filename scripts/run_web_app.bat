@echo off
setlocal DisableDelayedExpansion
pushd "%~dp0.."
if errorlevel 1 exit /b 2
set "ACTION=%~1"
if "%ACTION%"=="" set "ACTION=start"
if /i "%ACTION%"=="start" goto :start
if /i "%ACTION%"=="build" goto :build
if /i "%ACTION%"=="migrate" goto :migrate
if /i "%ACTION%"=="stop" goto :stop
if /i "%ACTION%"=="logs" goto :logs
echo Usage: %~nx0 [build^|migrate^|start^|stop^|logs]
set "RESULT=2"
goto :done
:build
docker build -t satellites-parcel-stats:local .
if errorlevel 1 goto :error
docker compose -f compose.web.yml build
goto :result
:migrate
docker compose -f compose.web.yml run --rm manage
goto :result
:start
docker compose -f compose.web.yml up -d api worker inspector web
if errorlevel 1 goto :error
echo Parcel Studio: http://localhost:8080
echo First setup requires: scripts\run_web_app.bat build, then scripts\run_web_app.bat migrate
goto :result
:stop
docker compose -f compose.web.yml stop
goto :result
:logs
docker compose -f compose.web.yml logs --tail 100 -f
goto :result
:error
set "RESULT=%ERRORLEVEL%"
echo Error: Web application command failed with code %RESULT%.
goto :done
:result
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" echo Error: Web application command failed with code %RESULT%.
:done
popd
endlocal & exit /b %RESULT%
