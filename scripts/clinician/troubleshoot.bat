@echo off
REM UWF Annotation Tool -- capture everything that goes wrong (Windows).
REM
REM Double-click this file. It writes uwf_troubleshoot.txt NEXT TO ITSELF
REM (same folder as this .bat), so there is no Desktop, no OneDrive and no
REM install path to get wrong. Send that file back.
REM
REM Deliberately linear: no loops, no delayed expansion, and every command
REM appends to the log, so a step that dies cannot take the log with it.

setlocal
set "LOG=%~dp0uwf_troubleshoot.txt"
set "INSTALL=%USERPROFILE%\uwf-annotate"

echo Writing report to:
echo   %LOG%
echo.
echo Please wait, this takes about a minute...
echo.

echo UWF annotation tool -- troubleshooting report > "%LOG%"
call :section "when"
date /t >> "%LOG%" 2>&1
time /t >> "%LOG%" 2>&1

call :section "where"
echo USERPROFILE=%USERPROFILE% >> "%LOG%"
echo this script=%~dp0 >> "%LOG%"
echo install guess=%INSTALL% >> "%LOG%"

call :section "install folder"
dir "%INSTALL%" >> "%LOG%" 2>&1

call :section "annotation_tool folder"
dir "%INSTALL%\annotation_tool" >> "%LOG%" 2>&1

call :section "batch folders"
dir /s /b "%INSTALL%\clinician_data" >> "%LOG%" 2>&1

call :section "conda environments on disk"
dir "%USERPROFILE%\miniconda3\envs" >> "%LOG%" 2>&1
dir "%USERPROFILE%\anaconda3\envs" >> "%LOG%" 2>&1
dir "%LOCALAPPDATA%\miniconda3\envs" >> "%LOG%" 2>&1
dir "%LOCALAPPDATA%\anaconda3\envs" >> "%LOG%" 2>&1
dir "C:\miniconda3\envs" >> "%LOG%" 2>&1
dir "C:\anaconda3\envs" >> "%LOG%" 2>&1

REM Locate the environment's python. Linear ifs on purpose - a for loop
REM here would need delayed expansion and is one more thing to get wrong.
set "ENVPY="
if exist "%USERPROFILE%\miniconda3\envs\uwf-annotate\python.exe" set "ENVPY=%USERPROFILE%\miniconda3\envs\uwf-annotate\python.exe"
if exist "%USERPROFILE%\anaconda3\envs\uwf-annotate\python.exe" set "ENVPY=%USERPROFILE%\anaconda3\envs\uwf-annotate\python.exe"
if exist "%USERPROFILE%\Miniconda3\envs\uwf-annotate\python.exe" set "ENVPY=%USERPROFILE%\Miniconda3\envs\uwf-annotate\python.exe"
if exist "%USERPROFILE%\Anaconda3\envs\uwf-annotate\python.exe" set "ENVPY=%USERPROFILE%\Anaconda3\envs\uwf-annotate\python.exe"
if exist "%LOCALAPPDATA%\miniconda3\envs\uwf-annotate\python.exe" set "ENVPY=%LOCALAPPDATA%\miniconda3\envs\uwf-annotate\python.exe"
if exist "%LOCALAPPDATA%\anaconda3\envs\uwf-annotate\python.exe" set "ENVPY=%LOCALAPPDATA%\anaconda3\envs\uwf-annotate\python.exe"
if exist "C:\miniconda3\envs\uwf-annotate\python.exe" set "ENVPY=C:\miniconda3\envs\uwf-annotate\python.exe"
if exist "C:\anaconda3\envs\uwf-annotate\python.exe" set "ENVPY=C:\anaconda3\envs\uwf-annotate\python.exe"

call :section "python"
if not defined ENVPY (
    echo NOT FOUND - the uwf-annotate environment does not exist. >> "%LOG%"
    echo Setup never finished building it. >> "%LOG%"
    goto :show
)
echo using: %ENVPY% >> "%LOG%"
"%ENVPY%" -V >> "%LOG%" 2>&1

call :section "packages"
"%ENVPY%" -c "import numpy, PIL, scipy, skimage; print('numpy', numpy.__version__); print('scipy', scipy.__version__); print('skimage', skimage.__version__)" >> "%LOG%" 2>&1

call :section "napari import"
"%ENVPY%" -c "import napari; print('napari', napari.__version__)" >> "%LOG%" 2>&1

call :section "full diagnostic (doctor.py)"
"%ENVPY%" "%INSTALL%\annotation_tool\doctor.py" "%~dp0doctor_report.txt" >> "%LOG%" 2>&1

call :section "opening a napari window - hardware graphics"
"%ENVPY%" -c "import napari, numpy as np; v = napari.Viewer(show=False); v.add_image(np.zeros((16,16))); print('VIEWER OK'); v.close()" >> "%LOG%" 2>&1

call :section "opening a napari window - software graphics"
set "QT_OPENGL=software"
set "LIBGL_ALWAYS_SOFTWARE=1"
"%ENVPY%" -c "import napari, numpy as np; v = napari.Viewer(show=False); v.add_image(np.zeros((16,16))); print('VIEWER OK WITH SOFTWARE GL'); v.close()" >> "%LOG%" 2>&1

:show
echo. >> "%LOG%"
echo === end of report === >> "%LOG%"
type "%LOG%"
echo.
echo ================================================================
echo   Report saved to:
echo     %LOG%
echo   Send that file to Lennert.
echo ================================================================
pause
endlocal
exit /b 0

:section
echo. >> "%LOG%"
echo ================================================================ >> "%LOG%"
echo == %~1 >> "%LOG%"
echo ================================================================ >> "%LOG%"
exit /b 0
