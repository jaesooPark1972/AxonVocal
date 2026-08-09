
Set WshShell = WScript.CreateObject("WScript.Shell")
Set oObj = WshShell.CreateShortcut("C:\Users\JayPark1004\Desktop\AXON_보컬렌더_Studio.lnk")
oObj.TargetPath = "d:\vocalRender\start_vocal_studio.bat"
oObj.WorkingDirectory = "d:\vocalRender"
oObj.WindowStyle = 1
oObj.IconLocation = "shell32.dll, 168"
oObj.Save
