@echo off
setlocal

set "FILTER_DLL=%~dp0PS3EyeSourceFilter.dll"
set "REGSVR32=%SystemRoot%\System32\regsvr32.exe"

rem Sysnative reaches the 64-bit system directory if this script is ever
rem launched from a 32-bit command prompt on 64-bit Windows.
if exist "%SystemRoot%\Sysnative\regsvr32.exe" set "REGSVR32=%SystemRoot%\Sysnative\regsvr32.exe"

if not exist "%FILTER_DLL%" (
    echo ERROR: PS3EyeSourceFilter.dll was not found next to this script.
    echo Keep all files from the package together in the same folder.
    pause
    exit /b 1
)

"%SystemRoot%\System32\whoami.exe" /groups | "%SystemRoot%\System32\findstr.exe" /c:"S-1-16-12288" >nul
if errorlevel 1 (
    echo ERROR: Administrator permission is required.
    echo Right-click this file and select "Run as administrator".
    pause
    exit /b 1
)

echo Registering the x64 PS3 Eye DirectShow filter...
"%REGSVR32%" "%FILTER_DLL%"
set "RESULT=%ERRORLEVEL%"

if not "%RESULT%"=="0" (
    echo.
    echo Registration failed with exit code %RESULT%.
    pause
    exit /b %RESULT%
)

echo.
echo Registration completed successfully.
pause
exit /b 0
