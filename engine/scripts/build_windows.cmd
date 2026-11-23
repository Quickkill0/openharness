@echo off
rem Build the OpenHarness engine on Windows with MSVC + Ninja (+ CUDA).
rem
rem   engine\scripts\build_windows.cmd [cuda|cpu] [extra cmake -D args...]
rem
rem Environment knobs:
rem   OH_CUDA_ARCH    CUDA architectures (default 89 = RTX 40xx)
rem   OH_BUILD_JOBS   parallel compile jobs (default 12)
rem   OH_BUILD_DIR    build directory (default engine\build-win-<variant>)
rem   OH_BUILD_TESTS  ON to build the engine and upstream tests (default ON)
setlocal enableextensions
rem Resolve before any shift: %~dp0 follows %0.
for %%i in ("%~dp0..") do set "ENGINE_DIR=%%~fi"

set "VARIANT=%~1"
if "%VARIANT%"=="" set "VARIANT=cuda"
shift
set "EXTRA="
:collect
if "%~1"=="" goto collected
set "EXTRA=%EXTRA% %1"
shift
goto collect
:collected

if "%OH_CUDA_ARCH%"=="" set "OH_CUDA_ARCH=89"
if "%OH_BUILD_JOBS%"=="" set "OH_BUILD_JOBS=12"
if "%OH_BUILD_TESTS%"=="" set "OH_BUILD_TESTS=ON"
if "%OH_BUILD_DIR%"=="" set "OH_BUILD_DIR=%ENGINE_DIR%\build-win-%VARIANT%"

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if not defined VSINSTALL (
    echo error: no Visual Studio with the C++ x64 tools found 1>&2
    exit /b 1
)
call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
rem vcvars puts the bundled CMake and Ninja on PATH.

set "CUDA_ARGS=-DGGML_CUDA=OFF"
if /i "%VARIANT%"=="cuda" set "CUDA_ARGS=-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=%OH_CUDA_ARCH%"

cmake -S "%ENGINE_DIR%" -B "%OH_BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DOH_ENGINE_BUILD_TESTS=%OH_BUILD_TESTS% %CUDA_ARGS% %EXTRA% || exit /b 1
cmake --build "%OH_BUILD_DIR%" --parallel %OH_BUILD_JOBS% || exit /b 1
echo built: %OH_BUILD_DIR%\bin\openharness-engine-server.exe
