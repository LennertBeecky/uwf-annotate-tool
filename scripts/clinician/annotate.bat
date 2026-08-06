@echo off
REM UWF Annotation Tool — start an annotation session (Windows).
REM Double-click to:
REM   1. find the latest batch zip dropped into clinician_data\incoming\
REM   2. extract images + predictions into the right per-batch folders
REM   3. launch the napari annotation tool with prefill on
REM Re-run any time during a batch — already-annotated images are skipped.

setlocal EnableDelayedExpansion

set "ENV_NAME=uwf-annotate"
set "ID_FILE=%USERPROFILE%\.uwf-annotate-id"

REM Silence noisy-but-harmless OpenMP / Qt warnings.
set "KMP_WARNINGS=0"
set "QT_LOGGING_RULES=qt.qpa.window.warning=false"

REM Force Python's basic REPL — Python 3.13's _pyrepl crashes on some
REM Windows consoles with WinError 123 when input() is called. The
REM uwf-annotate env pins python=3.11 so this is belt-and-braces in case
REM 'conda run' below somehow falls through to a different interpreter.
set "PYTHON_BASIC_REPL=1"

REM Resolve install dir = parent of parent of this script's location.
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%..\.." >nul
set "INSTALL_DIR=%CD%"
popd >nul
cd /d "%INSTALL_DIR%"

set "INCOMING=clinician_data\incoming"
set "PROCESSED=clinician_data\incoming\processed"
if not exist "%INCOMING%" mkdir "%INCOMING%"
if not exist "%PROCESSED%" mkdir "%PROCESSED%"

REM --- conda check — same logic as setup.bat: probe PATH then standard
REM     install locations, source activate.bat if needed.
set "CONDA_ACTIVATE="
where conda >nul 2>nul
if errorlevel 1 (
    for %%P in (
        "%USERPROFILE%\anaconda3\Scripts\activate.bat"
        "%USERPROFILE%\miniconda3\Scripts\activate.bat"
        "%USERPROFILE%\Anaconda3\Scripts\activate.bat"
        "%USERPROFILE%\Miniconda3\Scripts\activate.bat"
        "%LOCALAPPDATA%\anaconda3\Scripts\activate.bat"
        "%LOCALAPPDATA%\miniconda3\Scripts\activate.bat"
        "%PROGRAMDATA%\anaconda3\Scripts\activate.bat"
        "%PROGRAMDATA%\miniconda3\Scripts\activate.bat"
        "C:\anaconda3\Scripts\activate.bat"
        "C:\miniconda3\Scripts\activate.bat"
    ) do (
        if exist %%~P set "CONDA_ACTIVATE=%%~P"
    )
    if not defined CONDA_ACTIVATE (
        echo ERROR: conda not found. Re-run setup.bat first.
        pause & exit /b 1
    )
    call "!CONDA_ACTIVATE!"
)
REM Verify the env exists. We use 'conda run' below instead of
REM 'conda activate' because activate is unreliable inside batch
REM scripts on Windows (falls back to base python with wrong PATH).
call conda env list | findstr /B "%ENV_NAME% " >nul
if errorlevel 1 (
    echo ERROR: conda env '%ENV_NAME%' not found. Re-run setup.bat.
    pause & exit /b 1
)

REM --- annotator id
if not exist "%ID_FILE%" (
    echo ERROR: no annotator ID set. Re-run setup.bat first.
    pause & exit /b 1
)
set /p ANNOTATOR_ID=<"%ID_FILE%"
echo Annotator: %ANNOTATOR_ID%

REM --- find batch zip
set "ZIP="
for %%f in ("%INCOMING%\batch_*.zip") do (
    if not defined ZIP (
        set "ZIP=%%f"
    )
)

if defined ZIP (
    REM Extract the zip
    for %%n in ("!ZIP!") do set "BATCH_NAME=%%~nn"
    echo Extracting batch: !BATCH_NAME!

    REM Extraction is done by annotation_tool\extract_batch.py, not here:
    REM the shell versions drifted apart and the cmd one silently copied
    REM nothing when a zip arrived with a wrapper folder. The Python one is
    REM covered by tests/test_extract_batch.py.
    conda run --no-capture-output -n %ENV_NAME% python annotation_tool\extract_batch.py "!ZIP!" "%INSTALL_DIR%"
    if errorlevel 1 (
        echo.
        echo The batch could not be unpacked - see the message above.
        echo The zip is still in %INCOMING%, nothing was lost.
        pause & exit /b 1
    )


    move /Y "!ZIP!" "%PROCESSED%" >nul
    set "BATCH_DIR=clinician_data\images_to_annotate\!BATCH_NAME!"
) else (
    REM No new zip — pick latest in-progress batch
    set "BATCH_DIR="
    for /D %%d in (clinician_data\images_to_annotate\batch_*) do set "BATCH_DIR=%%d"
    if not defined BATCH_DIR (
        echo No batch zips and no in-progress batches found.
        echo Download batch_*.zip from OneDrive into %INSTALL_DIR%\%INCOMING%\
        echo and re-run this script.
        pause & exit /b 0
    )
    for %%n in ("!BATCH_DIR!") do set "BATCH_NAME=%%~nxn"
)

set "ANNOTATIONS_DIR=clinician_data\annotations\!BATCH_NAME!\%ANNOTATOR_ID%"
if not exist "!ANNOTATIONS_DIR!" mkdir "!ANNOTATIONS_DIR!"

echo.
echo ================================================================
echo   Batch:    !BATCH_NAME!
echo ================================================================
echo Launching napari... (close window or press 'q' to save+next, 's' to skip)
echo.

REM 'conda run --no-capture-output' is the recommended pattern for
REM non-interactive script invocation: it picks the env's interpreter
REM directly, regardless of activation state, and pipes stdout/stderr
REM through transparently (so napari output appears live).
REM Output is always filled pixel masks: the thickness you paint is the
REM vessel width. A centreline can be derived from that afterwards.
REM --prefill auto reads the batch to decide where the pre-filled vessels
REM come from: filled artery/vein masks (DVA) or <stem>_hard.png (UWF).
REM
REM Everything is logged. napari failing to start on Windows is usually a
REM graphics problem, and it dies without leaving anything on screen once
REM the window closes - so keep a file we can actually read afterwards.
set "LOGFILE=%INSTALL_DIR%\last_session_log.txt"

REM Software rendering FIRST on Windows. napari draws through OpenGL, and
REM remote desktop sessions and stock display drivers frequently provide no
REM usable GL context - the failure mode is the window never appearing. DVA
REM frames are 720x720, so software rasterising costs nothing noticeable.
REM If it fails we retry on the GPU below, which is the faster path for
REM large UWF images.
set "QT_OPENGL=software"
set "LIBGL_ALWAYS_SOFTWARE=1"

conda run --no-capture-output -n %ENV_NAME% python annotation_tool\annotate.py ^
    "!BATCH_DIR!" ^
    --output-dir "!ANNOTATIONS_DIR!" ^
    --prefill auto ^
    --masks-dir "clinician_data\predictions\!BATCH_NAME!" > "!LOGFILE!" 2>&1
set "RC=!ERRORLEVEL!"
type "!LOGFILE!"

if not "!RC!"=="0" (
    echo.
    echo ================================================================
    echo   That attempt failed ^(exit code !RC!^).
    echo   Retrying with the graphics card instead of software rendering.
    echo ================================================================
    echo.
    set "QT_OPENGL="
    set "LIBGL_ALWAYS_SOFTWARE="
    conda run --no-capture-output -n %ENV_NAME% python annotation_tool\annotate.py ^
        "!BATCH_DIR!" ^
        --output-dir "!ANNOTATIONS_DIR!" ^
        --prefill auto ^
        --masks-dir "clinician_data\predictions\!BATCH_NAME!" >> "!LOGFILE!" 2>&1
    set "RC=!ERRORLEVEL!"
    type "!LOGFILE!"
    if not "!RC!"=="0" (
        echo.
        echo ================================================================
        echo   Still failing. Opening the log in Notepad now - select all
        echo   ^(Ctrl+A^), copy ^(Ctrl+C^) and send it to Lennert.
        echo     !LOGFILE!
        echo ================================================================
        start "" notepad "!LOGFILE!"
    ) else (
        echo.
        echo   The graphics card worked. Tell Lennert - software rendering
        echo   can be turned off for this PC.
    )
)

echo.
echo Session ended.
echo When you're done with this batch, double-click upload.bat to package
echo your annotations for return to OneDrive.
pause
endlocal
