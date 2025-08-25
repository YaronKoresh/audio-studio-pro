@echo off
REM Audio Studio Pro

REM == Argument handler for elevated re-launch ==
if "%1"=="install_python" goto install_python_logic
if "%1"=="install_package_admin" goto install_package_as_admin

REM == Main script entry point ==
echo Checking for Python installation...
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Python not found. Administrator privileges are required to install it.
    goto install_python_prompt
)

echo Python found.
goto install_package_standard

:install_python_prompt
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Requesting administrative privileges...
    powershell -Command "Start-Process '%~f0' -Verb RunAs -ArgumentList 'install_python'"
    exit /b
)
goto install_python_logic

:install_python_logic
echo [INFO] Now running as Administrator. Attempting to download and install Python...
set "PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/3.11.5/python-3.11.5-amd64.exe"
set "INSTALLER_PATH=%TEMP%\python_installer.exe"
echo Downloading Python 3.11.5 installer...
powershell -Command "Invoke-WebRequest -Uri '%PYTHON_INSTALLER_URL%' -OutFile '%INSTALLER_PATH%'"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to download Python installer. Please check your internet connection.
    pause
    exit /b 1
)
echo Starting Python installation (this may take a few minutes)...
start /wait %INSTALLER_PATH% /quiet InstallAllUsers=1 PrependPath=1
del "%INSTALLER_PATH%"
echo [SUCCESS] Python has been installed.
echo Please re-run this script to continue with the Audio Studio Pro installation.
pause
exit /b 0

:install_package_standard
echo Installing/Updating Audio Studio Pro from GitHub with standard privileges...
python -m pip install --upgrade --force-reinstall git+https://github.com/YaronKoresh/audio-studio-pro.git
if %errorlevel% equ 0 (
    goto install_complete
)
echo [INFO] Installation failed, likely due to permissions. Retrying as Administrator...
powershell -Command "Start-Process '%~f0' -Verb RunAs -ArgumentList 'install_package_admin'"
exit /b

:install_package_as_admin
echo Now running as Administrator. Retrying installation...
python -m pip install --upgrade --force-reinstall git+https://github.com/YaronKoresh/audio-studio-pro.git
if %errorlevel% neq 0 (
    echo [ERROR] Installation failed even with administrator privileges.
    echo Please check your internet connection and pip setup.
    pause
    exit /b 1
)

:install_complete
echo Installation complete.
goto launch_app

:launch_app
echo Launching Audio Studio Pro...
echo.
audio-studio-pro

echo.
echo Audio Studio Pro has been closed.
pause