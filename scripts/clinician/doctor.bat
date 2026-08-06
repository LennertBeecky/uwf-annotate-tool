@echo off
REM UWF Annotation Tool — diagnostic (Windows).
REM Double-click when napari won't open. Writes a report into the install
REM folder; send that file to Lennert.

setlocal EnableDelayedExpansion
set "INSTALL_DIR=%USERPROFILE%\uwf-annotate"
REM Write into the install folder. The Desktop is not a safe target on
REM Windows: OneDrive redirects it, so %USERPROFILE%\Desktop often does
REM not exist and the report silently vanishes.
set "REPORT=%INSTALL_DIR%\uwf_annotate_diagnostic.txt"

echo ================================================================
echo   UWF Annotation Tool -- diagnostic
echo ================================================================
echo.

REM Run through 'conda run', NOT by calling envs\...\python.exe directly.
REM
REM This script used to do the latter, on the reasoning that activation was
REM unnecessary for a single script. That was wrong, and it sent us chasing
REM ghosts for hours. Calling the environment's python.exe directly starts
REM the right interpreter but does not activate the environment, which on
REM Windows leaves the environment's DLL folder off PATH. Qt then resolves
REM its native dependencies against whatever else is on PATH, loads a
REM mismatched DLL, and the process dies with 0xC06D007F the moment that
REM DLL is called -- long after the import that appeared to succeed.
REM
REM The same diagnostic passed under 'conda run' and crashed run directly,
REM on the same machine, minutes apart. So: conda run.
set "CONDA_EXE="
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if exist "%USERPROFILE%\Miniconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\Miniconda3\Scripts\conda.exe"
if exist "%USERPROFILE%\Anaconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\Anaconda3\Scripts\conda.exe"
if exist "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe" set "CONDA_EXE=%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"
if exist "%LOCALAPPDATA%\anaconda3\Scripts\conda.exe" set "CONDA_EXE=%LOCALAPPDATA%\anaconda3\Scripts\conda.exe"
if exist "%PROGRAMDATA%\miniconda3\Scripts\conda.exe" set "CONDA_EXE=%PROGRAMDATA%\miniconda3\Scripts\conda.exe"
if exist "%PROGRAMDATA%\anaconda3\Scripts\conda.exe" set "CONDA_EXE=%PROGRAMDATA%\anaconda3\Scripts\conda.exe"
if exist "C:\miniconda3\Scripts\conda.exe" set "CONDA_EXE=C:\miniconda3\Scripts\conda.exe"
if exist "C:\anaconda3\Scripts\conda.exe" set "CONDA_EXE=C:\anaconda3\Scripts\conda.exe"

if not defined CONDA_EXE where conda >nul 2>nul && set "CONDA_EXE=conda"

if not defined CONDA_EXE (
    echo ERROR: could not find conda.
    echo.
    echo That means setup did not finish. Re-run setup.bat and watch for
    echo errors while it builds the environment ^(the step that takes
    echo about 5 minutes^).
    echo.
    > "%REPORT%" echo conda not found - setup did not complete.
    echo A short report was still written to:
    echo   %REPORT%
    pause
    exit /b 1
)

echo   Using conda: !CONDA_EXE!
echo.
cd /d "%INSTALL_DIR%"
"!CONDA_EXE!" run --no-capture-output -n uwf-annotate python annotation_tool\doctor.py "%REPORT%"

echo.
echo ================================================================
echo   Report written to:
echo     %REPORT%
echo   Please send that file to Lennert.
echo ================================================================
pause
endlocal
