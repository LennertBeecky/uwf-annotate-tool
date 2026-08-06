@echo off
REM UWF Annotation Tool -- put napari back on the tested version (Windows).
REM
REM Double-click this when napari will not open. The install is not pinned
REM in older copies of this repo, so a PC set up on a different day can end
REM up with a napari the tool has never been tested against -- which is what
REM happened here: the tested version is 0.7.0, this PC resolved to 0.8.0,
REM and the viewer died with no error message at all.
REM
REM Writes uwf_repair_log.txt NEXT TO ITSELF. No Desktop, no OneDrive, no
REM install path to get wrong. Needs no admin rights.

setlocal
set "LOG=%~dp0uwf_repair_log.txt"
set "INSTALL=%USERPROFILE%\uwf-annotate"
set "WANTED=0.7.0"

echo ================================================================
echo   UWF Annotation Tool -- repair
echo ================================================================
echo.
echo Writing a log to:
echo   %LOG%
echo.

echo UWF annotation tool -- repair log > "%LOG%"
date /t >> "%LOG%" 2>&1
time /t >> "%LOG%" 2>&1

REM Find the environment's python directly. conda is usually not on PATH,
REM and we do not need activation to run pip.
set "ENVPY="
if exist "%USERPROFILE%\Miniconda3\envs\uwf-annotate\python.exe" set "ENVPY=%USERPROFILE%\Miniconda3\envs\uwf-annotate\python.exe"
if exist "%USERPROFILE%\miniconda3\envs\uwf-annotate\python.exe" set "ENVPY=%USERPROFILE%\miniconda3\envs\uwf-annotate\python.exe"
if exist "%USERPROFILE%\Anaconda3\envs\uwf-annotate\python.exe" set "ENVPY=%USERPROFILE%\Anaconda3\envs\uwf-annotate\python.exe"
if exist "%USERPROFILE%\anaconda3\envs\uwf-annotate\python.exe" set "ENVPY=%USERPROFILE%\anaconda3\envs\uwf-annotate\python.exe"
if exist "%LOCALAPPDATA%\miniconda3\envs\uwf-annotate\python.exe" set "ENVPY=%LOCALAPPDATA%\miniconda3\envs\uwf-annotate\python.exe"
if exist "%LOCALAPPDATA%\anaconda3\envs\uwf-annotate\python.exe" set "ENVPY=%LOCALAPPDATA%\anaconda3\envs\uwf-annotate\python.exe"
if exist "C:\miniconda3\envs\uwf-annotate\python.exe" set "ENVPY=C:\miniconda3\envs\uwf-annotate\python.exe"
if exist "C:\anaconda3\envs\uwf-annotate\python.exe" set "ENVPY=C:\anaconda3\envs\uwf-annotate\python.exe"

if not defined ENVPY (
    echo ERROR: could not find the uwf-annotate environment.
    echo ERROR: could not find the uwf-annotate environment. >> "%LOG%"
    echo Re-run setup.bat first.
    pause
    exit /b 1
)

echo Using: %ENVPY%
echo python=%ENVPY% >> "%LOG%"
echo.

echo Version before the repair: >> "%LOG%"
"%ENVPY%" -c "import napari; print(napari.__version__)" >> "%LOG%" 2>&1

echo Installing napari %WANTED% -- this takes a few minutes.
echo Please leave this window open until it says DONE.
echo.
echo --- pip install --------------------------------- >> "%LOG%"
"%ENVPY%" -m pip install "napari[all]==%WANTED%" >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"

echo. >> "%LOG%"
echo Version after the repair: >> "%LOG%"
"%ENVPY%" -c "import napari; print(napari.__version__)" >> "%LOG%" 2>&1

if not "%RC%"=="0" (
    echo.
    echo ================================================================
    echo   The install did not finish cleanly.
    echo   Send this file to Lennert:
    echo     %LOG%
    echo ================================================================
    start "" notepad "%LOG%"
    pause
    exit /b 1
)

echo.
echo --- checking that the viewer opens now ---------- >> "%LOG%"
cd /d "%INSTALL%"
"%ENVPY%" annotation_tool\doctor.py "%~dp0doctor_report_after_repair.txt" >> "%LOG%" 2>&1

echo.
echo ================================================================
echo   DONE.
echo   Now double-click annotate.bat and see if napari opens.
echo.
echo   If it still does not, send these two files to Lennert:
echo     %LOG%
echo     %~dp0doctor_report_after_repair.txt
echo ================================================================
pause
endlocal
