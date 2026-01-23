@echo off
REM build_libqex.bat - Wrapper to run PowerShell build script
REM
REM Prerequisites:
REM   - CMake (https://cmake.org/download/)
REM   - Visual Studio 2019+ with C++ workload
REM   - PowerShell (included in Windows 10+)
REM
REM Usage:
REM   scripts\build_libqex.bat

echo Running libQEx build script...
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0build_libqex.ps1"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Build failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Build succeeded!
pause
