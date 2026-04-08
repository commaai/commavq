$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location "$ScriptDir\.."

if (-Not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "Virtual environment not found. Please run scripts\setup_env.ps1 first."
    Read-Host -Prompt "Press Enter to exit"
    exit
}

Write-Host "Opening a new PowerShell session with the virtual environment activated..."
# Start a new PowerShell session that loads the activation script and stays open
powershell.exe -NoExit -Command ". '.\venv\Scripts\Activate.ps1'"
