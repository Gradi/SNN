@echo off

echo [%time%] Deleting old snn
python -m pip -q uninstall -y snn

if %errorlevel% NEQ 0 (
    echo [%time%] pip returned bad exit code.
)
echo [%time%] Installing snn
python -m pip install .

if %errorlevel% NEQ 0 (
    echo [%time%] pip returned bad exit code.
)
echo [%time%] Done!
timeout 5
