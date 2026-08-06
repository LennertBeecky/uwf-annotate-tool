@echo off
REM UWF Annotation Tool — diagnostic (Windows).
REM Double-click when napari won't open. Writes a report to your Desktop;
REM send that file to Lennert.

setlocal EnableDelayedExpansion
set "INSTALL_DIR=%USERPROFILE%\uwf-annotate"
REM OneDrive commonly redirects the Desktop, so %USERPROFILE%\Desktop may
REM not exist. Prefer the redirected one when it is there. doctor.py falls
REM back on its own too, and prints wherever it actually wrote.
set "REPORT=%USERPROFILE%\Desktop\uwf_annotate_diagnostic.txt"
if exist "%USERPROFILE%\OneDrive\Desktop" (
    set "REPORT=%USERPROFILE%\OneDrive\Desktop\uwf_annotate_diagnostic.txt"
)

echo ================================================================
echo   UWF Annotation Tool -- diagnostic
echo ================================================================
echo.

REM Find the environment's python directly: conda is usually not on PATH,
REM and we do not need activation to run one script.
set "ENVPY="
for %%P in (
    "%USERPROFILE%\miniconda3\envs\uwf-annotate\python.exe"
    "%USERPROFILE%\anaconda3\envs\uwf-annotate\python.exe"
    "%USERPROFILE%\Miniconda3\envs\uwf-annotate\python.exe"
    "%USERPROFILE%\Anaconda3\envs\uwf-annotate\python.exe"
    "%LOCALAPPDATA%\miniconda3\envs\uwf-annotate\python.exe"
    "%LOCALAPPDATA%\anaconda3\envs\uwf-annotate\python.exe"
    "%PROGRAMDATA%\miniconda3\envs\uwf-annotate\python.exe"
    "%PROGRAMDATA%\anaconda3\envs\uwf-annotate\python.exe"
    "C:\miniconda3\envs\uwf-annotate\python.exe"
    "C:\anaconda3\envs\uwf-annotate\python.exe"
) do (
    if exist %%~P set "ENVPY=%%~P"
)

if not defined ENVPY (
    echo ERROR: could not find the uwf-annotate environment.
    echo.
    echo That means setup did not finish. Re-run setup.bat and watch for
    echo errors while it builds the environment ^(the step that takes
    echo about 5 minutes^).
    echo.
    > "%REPORT%" echo uwf-annotate environment not found - setup did not complete.
    echo A short report was still written to:
    echo   %REPORT%
    pause
    exit /b 1
)

echo   Using: !ENVPY!
echo.
cd /d "%INSTALL_DIR%"
"!ENVPY!" annotation_tool\doctor.py "%REPORT%"

echo.
echo ================================================================
echo   Report written to:
echo     %REPORT%
echo   Please send that file to Lennert.
echo ================================================================
pause
endlocal
