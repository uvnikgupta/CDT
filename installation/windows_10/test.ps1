Write-Host "Conda env name for cdt installation: " -ForegroundColor Green -NoNewline
$condaenv = Read-Host

throw $condaenv

Write-Output "continue here"