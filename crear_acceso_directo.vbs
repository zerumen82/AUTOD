Set WshShell = CreateObject("WScript.Shell")
Set Shortcut = WshShell.CreateShortcut(WshShell.SpecialFolders("Desktop") & "\AutoAuto.lnk")
Shortcut.TargetPath = "D:\PROJECTS\AUTOAUTO\AutoAuto_Edit.bat"
Shortcut.WorkingDirectory = "D:\PROJECTS\AUTOAUTO"
Shortcut.Description = "AutoAuto Image Editor con fixes aplicados"
Shortcut.Save