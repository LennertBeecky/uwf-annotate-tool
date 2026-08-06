@echo off
REM UWF Annotation Tool -- rebuild the environment from scratch (Windows).
REM
REM Use this when the environment is in a broken half-state: packages that
REM used to import now crash, or a pip install stopped part way through.
REM Installing on top of a broken environment does not repair it -- pip
REM leaves whatever it already changed in place. Deleting and rebuilding is
REM the only reliable route back to a known-good set of packages.
REM
REM Nothing you have annotated is touched: annotations live in
REM clinician_data\, the environment does not.
REM
REM Writes uwf_rebuild_log.txt NEXT TO ITSELF. Takes about 10 minutes.

setlocal
set "LOG=%~dp0uwf_rebuild_log.txt"
set "INSTALL=%USERPROFILE%\uwf-annotate"
set "ENV_NAME=uwf-annotate"

echo ================================================================
echo   UWF Annotation Tool -- rebuild the environment
echo ================================================================
echo.
echo This deletes and recreates the '%ENV_NAME%' environment.
echo Your annotations are NOT affected.
echo.
echo Log file:
echo   %LOG%
echo.

echo UWF annotation tool -- rebuild log > "%LOG%"
date /t >> "%LOG%" 2>&1
time /t >> "%LOG%" 2>&1

REM --- locate conda. It is usually not on PATH; source activate.bat so the
REM     'conda' command works for the rest of this script.
where conda >nul 2>nul
if errorlevel 1 (
    set "CONDA_ACTIVATE="
    if exist "%USERPROFILE%\Miniconda3\Scripts\activate.bat" set "CONDA_ACTIVATE=%USERPROFILE%\Miniconda3\Scripts\activate.bat"
    if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" set "CONDA_ACTIVATE=%USERPROFILE%\miniconda3\Scripts\activate.bat"
    if exist "%USERPROFILE%\Anaconda3\Scripts\activate.bat" set "CONDA_ACTIVATE=%USERPROFILE%\Anaconda3\Scripts\activate.bat"
    if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" set "CONDA_ACTIVATE=%USERPROFILE%\anaconda3\Scripts\activate.bat"
    if exist "%LOCALAPPDATA%\miniconda3\Scripts\activate.bat" set "CONDA_ACTIVATE=%LOCALAPPDATA%\miniconda3\Scripts\activate.bat"
    if exist "C:\miniconda3\Scripts\activate.bat" set "CONDA_ACTIVATE=C:\miniconda3\Scripts\activate.bat"
    if not defined CONDA_ACTIVATE (
        echo ERROR: conda not found. >> "%LOG%"
        echo ERROR: conda not found. Re-run setup.bat first.
        pause
        exit /b 1
    )
    call "%CONDA_ACTIVATE%"
)

cd /d "%INSTALL%"
echo install dir=%INSTALL% >> "%LOG%"

echo --- conda info ---------------------------------- >> "%LOG%"
call conda info >> "%LOG%" 2>&1

echo --- packages before the rebuild ----------------- >> "%LOG%"
call conda list -n %ENV_NAME% >> "%LOG%" 2>&1

echo [1/3] Removing the old environment...
echo --- conda env remove ---------------------------- >> "%LOG%"
call conda env remove -n %ENV_NAME% -y >> "%LOG%" 2>&1

echo [2/3] Building a fresh one -- this is the slow part, about 10 minutes.
echo       Leave this window open.
echo --- conda env create ---------------------------- >> "%LOG%"
call conda env create -f environment_clinician.yml >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo. >> "%LOG%"
    echo conda env create FAILED with code %RC% >> "%LOG%"
    echo.
    echo ================================================================
    echo   The rebuild did not finish. Send this file to Lennert:
    echo     %LOG%
    echo ================================================================
    start "" notepad "%LOG%"
    pause
    exit /b 1
)

echo --- packages after the rebuild ------------------ >> "%LOG%"
call conda list -n %ENV_NAME% >> "%LOG%" 2>&1

echo [3/3] Checking the result...
echo --- doctor -------------------------------------- >> "%LOG%"
call conda run --no-capture-output -n %ENV_NAME% python annotation_tool\doctor.py "%~dp0doctor_report.txt" >> "%LOG%" 2>&1

echo.
echo ================================================================
echo   DONE. Now double-click annotate.bat.
echo.
echo   If napari still does not open, send BOTH files to Lennert:
echo     %LOG%
echo     %~dp0doctor_report.txt
echo ================================================================
pause
endlocal
