Set-Location 'D:\PROJECTS\AUTOAUTO'

# Undo last commit but keep working tree changes
git reset HEAD~1 --quiet

# Get changed files
$changed = git diff --name-only

# Exclude PNGs and test images, include code only
$pngs = $changed | Where-Object { $_ -like 'img_editor/*.png' -or $_ -like 'testdata/*.jpg' }
$code = $changed | Where-Object { $_ -notlike 'img_editor/*.png' -and $_ -notlike 'testdata/*.jpg' -and $_ -ne '_git_check.ps1' }

Write-Host "PNGs excluded:" $pngs.Count
Write-Host "Code files:" $code.Count

foreach ($f in $code) {
    git add $f
}

$stat = git diff --cached --stat
Write-Host $stat
Write-Host ("Total staged: " + ($stat | Measure-Object -Property Insertions,Deletions -Sum).Sum)
