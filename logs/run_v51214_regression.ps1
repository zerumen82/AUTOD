$env:PYTHONWARNINGS='ignore'
$py = 'D:\PROJECTS\AUTOAUTO\venv\Scripts\python.exe'
$probes = @('probe_editor_recolor_part_dispatch','probe_editor_compound_fast_dispatch','probe_editor_wall_instructions_20261015','probe_editor_measured_paste_zone','probe_editor_second_edit_dispatch','probe_editor_removal_paste_silhouette_20261015','probe_editor_body_clause_guard','probe_editor_paste_guard','probe_editor_change_scope','probe_editor_prompt_not_applied')
$out = 'D:\PROJECTS\AUTOAUTO\logs\v51214_regression.txt'
"START $(Get-Date -Format s)" | Set-Content $out
foreach ($p in $probes) {
  & $py -m diagnostics.$p *>> $out
  "RESULT $p EXIT=$LASTEXITCODE" | Add-Content $out
}
"DONE $(Get-Date -Format s)" | Add-Content $out
