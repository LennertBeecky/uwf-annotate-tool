@echo off
REM UWF Annotation Tool — one-time setup for clinicians (Windows).
REM Double-click this file to install / update the annotation tool.
REM Idempotent: re-run any time to update.

setlocal EnableDelayedExpansion

set "INSTALL_DIR=%USERPROFILE%\uwf-annotate"
set "REPO_URL=https://github.com/LennertBeecky/uwf-annotate-tool.git"
set "BRANCH=main"
set "ENV_NAME=uwf-annotate"
set "ID_FILE=%USERPROFILE%\.uwf-annotate-id"

echo ================================================================
echo   UWF Vessel Annotation Tool -- Setup
echo ================================================================
echo.

REM 1. conda check
where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: conda was not found on your PATH.
    echo.
    echo Please install miniconda first:
    echo   https://docs.conda.io/en/latest/miniconda.html
    echo.
    echo Choose the Windows installer and run it with default settings.
    echo Then open a new "Anaconda Prompt" or restart your computer
    echo and re-run this setup script.
    pause
    exit /b 1
)
echo   conda found.

REM 2. Repo: clone or pull
if exist "%INSTALL_DIR%\.git" (
    echo   Updating existing install at %INSTALL_DIR% ...
    pushd "%INSTALL_DIR%"
    git fetch --all --quiet
    git checkout %BRANCH% --quiet
    git pull --quiet
    popd
) else (
    echo   Cloning repository to %INSTALL_DIR% ...
    git clone --quiet %REPO_URL% "%INSTALL_DIR%"
    pushd "%INSTALL_DIR%"
    git checkout %BRANCH% --quiet
    popd
)

REM 3. Conda env
call conda env list | findstr /B "%ENV_NAME% " >nul
if errorlevel 1 (
    echo   Creating conda environment %ENV_NAME% ^(this takes ~5 min^) ...
    call conda env create -n %ENV_NAME% -f "%INSTALL_DIR%\environment_clinician.yml" --quiet
) else (
    echo   Updating conda environment %ENV_NAME% ...
    call conda env update -n %ENV_NAME% -f "%INSTALL_DIR%\environment_clinician.yml" --prune --quiet
)

REM 4. Annotator ID prompt (one-time)
if not exist "%ID_FILE%" (
    echo.
    echo   Setting your annotator ID ^(used to label your annotations^).
    set /p "ANNOTATOR=  Enter your name (lowercase, no spaces, e.g. ingrid): "
    if "!ANNOTATOR!"=="" (
        set "ANNOTATOR=anonymous"
        echo   No name entered -- defaulting to 'anonymous'.
    )
    >"%ID_FILE%" echo !ANNOTATOR!
    echo   Saved annotator ID: !ANNOTATOR!
) else (
    set /p ANNOTATOR=<"%ID_FILE%"
    echo   Annotator ID: !ANNOTATOR! ^(from %ID_FILE%^)
)

REM 5. Data folders
if not exist "%INSTALL_DIR%\clinician_data\incoming" (
    mkdir "%INSTALL_DIR%\clinician_data\incoming"
)
if not exist "%INSTALL_DIR%\clinician_data\incoming\processed" (
    mkdir "%INSTALL_DIR%\clinician_data\incoming\processed"
)

echo.
echo ================================================================
echo   Setup complete!
echo ================================================================
echo.
echo   Install location: %INSTALL_DIR%
echo.
echo   Next steps:
echo     1. Download a batch zip ^(e.g. batch_2026-04-30.zip^) from OneDrive.
echo     2. Move it into:
echo          %INSTALL_DIR%\clinician_data\incoming\
echo     3. Double-click annotate.bat ^(in scripts\clinician\^) to start.
echo.
echo   When you're done with a batch, double-click upload.bat to package
echo   your annotations for upload to OneDrive.
echo.
pause
endlocal
