@echo off
REM UWF Annotation Tool — package annotations for upload (Windows).

setlocal EnableDelayedExpansion

set "ID_FILE=%USERPROFILE%\.uwf-annotate-id"

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%..\.." >nul
set "INSTALL_DIR=%CD%"
popd >nul
cd /d "%INSTALL_DIR%"

if not exist "%ID_FILE%" (
    echo ERROR: no annotator ID set. Re-run setup.bat first.
    pause & exit /b 1
)
set /p ANNOTATOR_ID=<"%ID_FILE%"

set "ANNOTATIONS_ROOT=clinician_data\annotations"
if not exist "%ANNOTATIONS_ROOT%" (
    echo No annotations folder found.
    pause & exit /b 0
)

set /a PACKED=0
for /D %%b in ("%ANNOTATIONS_ROOT%\batch_*") do (
    set "BATCH_DIR=%%b"
    for %%n in ("!BATCH_DIR!") do set "BATCH_NAME=%%~nxn"
    set "ANNOTATOR_DIR=!BATCH_DIR!\%ANNOTATOR_ID%"
    if exist "!ANNOTATOR_DIR!" (
        set /a N=0
        for %%f in ("!ANNOTATOR_DIR!\*_artery.png") do set /a N+=1
        if !N! GTR 0 (
            set "TAR_NAME=!BATCH_NAME!_%ANNOTATOR_ID%_annotations.zip"
            echo Packing !TAR_NAME! ^(!N! annotated images^) ...
            REM Use built-in PowerShell to make a .zip (no tar.gz on stock Windows).
            powershell -Command "Compress-Archive -Force -Path '!ANNOTATOR_DIR!\*' -DestinationPath '%ANNOTATIONS_ROOT%\!TAR_NAME!'"
            set /a PACKED+=1
        )
    )
)

if !PACKED! EQU 0 (
    echo Nothing to pack -- no annotations from %ANNOTATOR_ID% found.
    pause & exit /b 0
)

echo.
echo ================================================================
echo   Packed !PACKED! batch^(es^). Files ready to upload:
echo ================================================================
dir /b "%ANNOTATIONS_ROOT%\*.zip"
echo.
echo   Drag each .zip onto your OneDrive 'returned\%ANNOTATOR_ID%\' folder.
echo   The Explorer window opening below has the files.
echo.

start "" "%ANNOTATIONS_ROOT%"
pause
endlocal
