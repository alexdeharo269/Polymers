# make.ps1
$BASH = "C:\msys64\usr\bin\bash.exe"
$PATH_WIN = (Get-Location).Path
$PATH_MSYS = & $BASH -lc "cd '$PATH_WIN'; pwd -W"

& $BASH -lc "cd '$PATH_MSYS' && make $($args -join ' ')"
