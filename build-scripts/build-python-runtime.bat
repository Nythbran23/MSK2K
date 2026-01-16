@echo off
REM Build Python venv for Windows

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set PYTHON_DIR=%PROJECT_ROOT%\python
set RESOURCES_DIR=%PROJECT_ROOT%\resources
set PLATFORM=win-x64

echo Building Python venv for platform: %PLATFORM%

REM Create resources directory structure
set RUNTIME_DIR=%RESOURCES_DIR%\python-runtime\%PLATFORM%
if not exist "%RUNTIME_DIR%" mkdir "%RUNTIME_DIR%"

REM Clean old venv if exists
if exist "%RUNTIME_DIR%\Scripts" (
    echo Cleaning old venv...
    rmdir /s /q "%RUNTIME_DIR%"
    mkdir "%RUNTIME_DIR%"
)

REM Create fresh venv
echo Creating Python virtual environment...
python -m venv "%RUNTIME_DIR%"

set PYTHON_EXE=%RUNTIME_DIR%\Scripts\python.exe
set PIP_EXE=%RUNTIME_DIR%\Scripts\pip.exe

REM Upgrade pip
echo Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip

REM Install requirements
echo Installing Python packages...
"%PIP_EXE%" install -r "%PYTHON_DIR%\requirements.txt"

REM Copy application files to venv
echo Copying application files...
set APP_DIR=%RUNTIME_DIR%\app
if not exist "%APP_DIR%" mkdir "%APP_DIR%"
copy "%PYTHON_DIR%\msk2k_audio_qso_server_Q12.py" "%APP_DIR%\"
copy "%PYTHON_DIR%\msk2k_complete.py" "%APP_DIR%\"
copy "%PYTHON_DIR%\msk2k_audio_qso_ui_Q12.html" "%APP_DIR%\"

REM Create version file
echo v0.2.0> "%RUNTIME_DIR%\runtime_version.txt"

REM List installed packages
echo Installed packages:
"%PIP_EXE%" list

echo.
echo ✅ Python venv built successfully!
echo    Platform: %PLATFORM%
echo    Location: %RUNTIME_DIR%
echo.

endlocal
