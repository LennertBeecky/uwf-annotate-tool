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

REM --- conda check
where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: conda not found. Re-run setup.bat first.
    pause & exit /b 1
)
call conda activate %ENV_NAME% 2>nul
if errorlevel 1 (
    echo ERROR: failed to activate conda env %ENV_NAME%. Re-run setup.bat.
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
    if not exist "clinician_data\images_to_annotate\!BATCH_NAME!" mkdir "clinician_data\images_to_annotate\!BATCH_NAME!"
    if not exist "clinician_data\predictions\!BATCH_NAME!" mkdir "clinician_data\predictions\!BATCH_NAME!"

    set "TMPDIR=%TEMP%\uwf_extract_!BATCH_NAME!"
    if exist "!TMPDIR!" rmdir /s /q "!TMPDIR!"
    mkdir "!TMPDIR!"
    powershell -Command "Expand-Archive -Path '!ZIP!' -DestinationPath '!TMPDIR!' -Force"

    if exist "!TMPDIR!\images" (
        xcopy /Y /Q "!TMPDIR!\images\*" "clinician_data\images_to_annotate\!BATCH_NAME!\" >nul
    )
    if exist "!TMPDIR!\predictions" (
        xcopy /Y /Q "!TMPDIR!\predictions\*" "clinician_data\predictions\!BATCH_NAME!\" >nul
    )

    move /Y "!ZIP!" "%PROCESSED%\" >nul
    rmdir /s /q "!TMPDIR!"
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

python annotation_tool\annotate.py ^
    "!BATCH_DIR!\" ^
    --output-dir "!ANNOTATIONS_DIR!\" ^
    --prefill predictions ^
    --predictions-dir "clinician_data\predictions\!BATCH_NAME!\"

echo.
echo Session ended.
echo When you're done with this batch, double-click upload.bat to package
echo your annotations for return to OneDrive.
pause
endlocal
