# Starts MindMesh at http://127.0.0.1:8000  (run from anywhere: .\run.ps1)
& "$PSScriptRoot\.venv\Scripts\python.exe" -m uvicorn mindmesh.api:app --app-dir "$PSScriptRoot\backend" --port 8000
