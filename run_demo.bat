@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1
title Hybrid Cognitive AI Pipeline
cd /d "%~dp0"

rem --- Locate a Python interpreter -------------------------------------------
set "PY_CMD="
where python >nul 2>&1
if %errorlevel%==0 set "PY_CMD=python"
if not defined PY_CMD (
    where python3 >nul 2>&1
    if %errorlevel%==0 set "PY_CMD=python3"
)
if not defined PY_CMD (
    where py >nul 2>&1
    if %errorlevel%==0 set "PY_CMD=py"
)
if not defined PY_CMD (
    echo.
    echo   Could not find Python on your PATH.
    echo   Install Python 3.10+ from https://www.python.org/downloads/
    echo   and make sure "Add python.exe to PATH" is checked during setup.
    echo.
    pause
    exit /b 1
)

if not exist "demo_showcase.py" (
    echo.
    echo   demo_showcase.py wasn't found next to this .bat file.
    echo   Make sure run_demo.bat stays in the project's root folder.
    echo.
    pause
    exit /b 1
)

:menu
cls
echo ============================================================
echo   Hybrid Cognitive AI Pipeline -- Quick Launcher
echo   (using: !PY_CMD!)
echo ============================================================
echo.
echo   1. Run the showcase demo         (~10s,  4 real scenes)
echo   2. Showcase demo + full test suite (~45s, most rigorous)
echo   3. Ask the pipeline your own question
echo   4. Run the full test suite only  (137 + 24 checks)
echo   5. Open interview prep notes
echo   6. Exit
echo.
set /p choice="  Choose an option (1-6): "

if "!choice!"=="1" goto demo
if "!choice!"=="2" goto demo_tests
if "!choice!"=="3" goto ask
if "!choice!"=="4" goto tests
if "!choice!"=="5" goto notes
if "!choice!"=="6" goto end
goto menu

:demo
echo.
!PY_CMD! demo_showcase.py
echo.
pause
goto menu

:demo_tests
echo.
!PY_CMD! demo_showcase.py --run-tests
echo.
pause
goto menu

:ask
echo.
echo   (press Enter with no text to use the default example question)
set /p question="  Your question: "
if "!question!"=="" set "question=Solve the integral of 2x from 0 to 4."
echo.
!PY_CMD! hybrid_cli.py --mode pipeline --text "!question!"
echo.
pause
goto menu

:tests
echo.
!PY_CMD! tests\test_smoke.py
echo.
!PY_CMD! -m pytest -q
echo.
pause
goto menu

:notes
if exist "docs\INTERVIEW_PREP.md" (
    start "" notepad "docs\INTERVIEW_PREP.md"
) else (
    echo.
    echo   docs\INTERVIEW_PREP.md not found.
    echo.
    pause
)
goto menu

:end
endlocal
exit /b 0
