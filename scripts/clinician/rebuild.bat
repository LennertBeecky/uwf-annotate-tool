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

REM --- Locate conda.exe and call it by full path.
REM
REM Every one of these probes is a separate top-level line on purpose. An
REM earlier version put them inside an "if errorlevel 1 ( ... )" block and
REM then used %CONDA_EXE% in the same block: cmd expands %VAR% when it
REM PARSES a block, not when it runs it, so the variable was still empty at
REM that point and the script reported "conda is not recognized". Keeping
REM the probes at top level side-steps delayed expansion completely.
REM
REM Calling conda.exe directly also means we never have to get 'conda
REM activate' working inside a batch script, which is its own minefield.
set "CONDA_EXE="
if exist "%USERPROFILE%\Miniconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\Miniconda3\Scripts\conda.exe"
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if exist "%USERPROFILE%\Anaconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\Anaconda3\Scripts\conda.exe"
if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if exist "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe" set "CONDA_EXE=%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"
if exist "%LOCALAPPDATA%\anaconda3\Scripts\conda.exe" set "CONDA_EXE=%LOCALAPPDATA%\anaconda3\Scripts\conda.exe"
if exist "%PROGRAMDATA%\miniconda3\Scripts\conda.exe" set "CONDA_EXE=%PROGRAMDATA%\miniconda3\Scripts\conda.exe"
if exist "%PROGRAMDATA%\anaconda3\Scripts\conda.exe" set "CONDA_EXE=%PROGRAMDATA%\anaconda3\Scripts\conda.exe"
if exist "C:\miniconda3\Scripts\conda.exe" set "CONDA_EXE=C:\miniconda3\Scripts\conda.exe"
if exist "C:\anaconda3\Scripts\conda.exe" set "CONDA_EXE=C:\anaconda3\Scripts\conda.exe"

REM Last resort: if conda happens to be on PATH, use it as a bare command.
if not defined CONDA_EXE where conda >nul 2>nul && set "CONDA_EXE=conda"

if not defined CONDA_EXE goto :no_conda

echo Using conda: %CONDA_EXE%
echo conda=%CONDA_EXE% >> "%LOG%"
echo.

cd /d "%INSTALL%"
echo install dir=%INSTALL% >> "%LOG%"

REM The pinned dependency list must be the current one, or the rebuild will
REM happily reinstall the same untested versions we are trying to leave.
echo --- environment_clinician.yml in use ------------ >> "%LOG%"
type environment_clinician.yml >> "%LOG%" 2>&1

echo --- conda info ---------------------------------- >> "%LOG%"
"%CONDA_EXE%" info >> "%LOG%" 2>&1

echo --- packages before the rebuild ----------------- >> "%LOG%"
"%CONDA_EXE%" list -n %ENV_NAME% >> "%LOG%" 2>&1

echo [1/3] Removing the old environment...
echo --- conda env remove ---------------------------- >> "%LOG%"
"%CONDA_EXE%" env remove -n %ENV_NAME% -y >> "%LOG%" 2>&1

REM conda leaves the folder behind if any file in it was locked (an editor,
REM an antivirus scan, a stray python.exe). A leftover folder makes the
REM create below fail with "prefix already exists", so clear it.
set "ENV_DIR="
for /f "usebackq delims=" %%D in (`"%CONDA_EXE%" info --base 2^>nul`) do set "ENV_DIR=%%D\envs\%ENV_NAME%"
if defined ENV_DIR if exist "%ENV_DIR%" (
    echo       (clearing leftover folder^)
    echo removing leftover %ENV_DIR% >> "%LOG%"
    rmdir /s /q "%ENV_DIR%" >> "%LOG%" 2>&1
)

echo [2/3] Building a fresh one -- this is the slow part, about 10 minutes.
echo       Leave this window open.
echo --- conda env create ---------------------------- >> "%LOG%"
"%CONDA_EXE%" env create -f environment_clinician.yml >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" goto :create_failed

echo --- packages after the rebuild ------------------ >> "%LOG%"
"%CONDA_EXE%" list -n %ENV_NAME% >> "%LOG%" 2>&1

echo [3/3] Checking the result...
echo --- doctor -------------------------------------- >> "%LOG%"
"%CONDA_EXE%" run --no-capture-output -n %ENV_NAME% python annotation_tool\doctor.py "%~dp0doctor_report.txt" >> "%LOG%" 2>&1

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
exit /b 0

:no_conda
echo ERROR: conda not found. >> "%LOG%"
echo.
echo ERROR: could not find conda.exe in any of the usual places.
echo.
echo Please send this file to Lennert:
echo   %LOG%
echo.
echo It would also help to run this and send what it prints:
echo   dir /b "%USERPROFILE%\Miniconda3\Scripts\conda.exe"
echo.
pause
exit /b 1

:create_failed
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
