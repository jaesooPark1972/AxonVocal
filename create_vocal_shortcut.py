import os
import subprocess

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
target_bat = r"d:\vocalRender\start_vocal_studio.bat"
shortcut_path = os.path.join(desktop_path, "AXON_보컬렌더_Studio.lnk")

vbs_script = f"""
Set WshShell = WScript.CreateObject("WScript.Shell")
Set oObj = WshShell.CreateShortcut("{shortcut_path}")
oObj.TargetPath = "{target_bat}"
oObj.WorkingDirectory = "d:\\vocalRender"
oObj.WindowStyle = 1
oObj.IconLocation = "shell32.dll, 168"
oObj.Save
"""

vbs_file = os.path.join(os.path.dirname(__file__), "create_vocal_shortcut.vbs")
with open(vbs_file, "w", encoding="utf-8") as f:
    f.write(vbs_script)

subprocess.run(["cscript", "//Nologo", vbs_file], check=True)
print("Desktop shortcut for VocalStudio successfully created!")
