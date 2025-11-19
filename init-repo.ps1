<#
  init-repo.ps1
  Usage: Open PowerShell in your project folder and run:
         .\init-repo.ps1 -RemoteUrl "https://github.com/you/repo.git" -Private:$false
  If you have the GitHub CLI (gh) installed, the script will try to create the remote repo for you.
#>

param(
  [string]$RemoteUrl = "",
  [string]$RepoName = "",
  [string]$Description = "My project",
  [bool]$Private = $false
)

function Write-Ok { Write-Host "✔ $args" -ForegroundColor Green }
function Write-Warn { Write-Host "⚠ $args" -ForegroundColor Yellow }
function Write-Err { Write-Host "✖ $args" -ForegroundColor Red }

# 0) quick checks
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  Write-Err "git is not installed or not in PATH. Install Git for Windows first."
  exit 1
}

# Optional: check gh (GitHub CLI)
$hasGh = (Get-Command gh -ErrorAction SilentlyContinue) -ne $null

# 1) create README, LICENSE, .gitignore if missing
if (-not (Test-Path README.md)) {
  "## $($RepoName -ne '' ? $RepoName : (Split-Path -Leaf (Get-Location)))`n`n$Description" | Out-File -Encoding utf8 README.md
  Write-Ok "Created README.md"
}
if (-not (Test-Path LICENSE)) {
  $year = (Get-Date).Year
  $lic = @"
MIT License

Copyright (c) $year

Permission is hereby granted, free of charge, to any person obtaining a copy...
(Shortened for script — replace with full MIT text if desired)
"@
  $lic | Out-File -Encoding utf8 LICENSE
  Write-Ok "Created LICENSE (MIT stub). Replace with full text if desired."
}
if (-not (Test-Path .gitignore)) {
  @(
    "# build artifacts",
    "*.exe",
    "*.o",
    "*.obj",
    "*.log",
    "*.out",
    "# Visual Studio / IDE",
    ".vs/",
    "*.user",
    "# Python/visuals",
    "__pycache__/",
    "*.py[cod]",
    "# MSYS2 / mingw caches",
    "/mingw64/bin/"
  ) | Out-File -Encoding utf8 .gitignore
  Write-Ok "Created .gitignore"
}

# 2) init git if needed
if (-not (Test-Path .git\HEAD)) {
  git init
  Write-Ok "Initialized git repository"
} else {
  Write-Warn "Git repo already initialized"
}

# 3) initial commit (if no commits)
$hasCommits = (& git rev-parse --verify HEAD 2>$null) -ne $null
if (-not $hasCommits) {
  git add -A
  git commit -m "chore: initial commit (README + LICENSE + .gitignore)"
  Write-Ok "Created initial commit"
} else {
  Write-Warn "Repository already has commits"
}

# 4) create remote via gh if available, otherwise use RemoteUrl param
if ($hasGh) {
  Write-Host "gh detected — attempting to create remote repo (non-interactive)..."
  $ghVisibility = $Private ? "--private" : "--public"
  # If RepoName not provided, gh will default to current directory name when using --source=.
  $createCmd = "gh repo create $($RepoName) $ghVisibility --source=. --description `"$Description`" --push --confirm"
  Write-Host $createCmd
  try {
    & gh repo create $RepoName @($ghVisibility) --source=. --description $Description --push --confirm
    Write-Ok "Created remote on GitHub and pushed commits (via gh)."
    exit 0
  } catch {
    Write-Warn "gh create failed or was interactive — falling back to RemoteUrl (if provided)."
  }
}

# 5) fallback: if user gave a remote URL, add and push
if ($RemoteUrl -ne "") {
  # set origin (overwrite if exists)
  $existing = git remote get-url origin 2>$null
  if ($lastExitCode -eq 0) {
    Write-Warn "Remote 'origin' already exists at $existing — updating to $RemoteUrl"
    git remote set-url origin $RemoteUrl
  } else {
    git remote add origin $RemoteUrl
    Write-Ok "Added origin -> $RemoteUrl"
  }
  # push main or master depending on branch
  $currentBranch = (& git symbolic-ref --quiet --short HEAD) -replace "`n",""
  if (-not $currentBranch) { $currentBranch = "main" }
  # ensure branch exists locally
  git branch --show-current 2>$null | Out-Null
  git push -u origin $currentBranch
  Write-Ok "Pushed branch '$currentBranch' to origin"
} else {
  Write-Warn "No remote created. To create on GitHub manually, run: gh repo create --source=. --public --push or create repo on github.com and then run: git remote add origin <url> ; git push -u origin main"
}

Write-Host "`nDone."
