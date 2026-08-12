import subprocess
from tools.filesystem import FileSystem

class PythonSandbox:
    def __init__(self, project_id: int):
        self.project_id = project_id
        self.fs = FileSystem(project_id)
        self.image = "research-sandbox"

    def run(self, code: str) -> dict:
        self.fs.write("workspace/generated.py", code)

        host_path = self.fs.base_path.resolve()

        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--network", "none",
                "--memory", "512m",
                "--pids-limit", "100",
                "-v", f"{host_path}/sources:/workspace/sources:ro",
                "-v", f"{host_path}/workspace:/workspace/workspace:rw",
                "-v", f"{host_path}/outputs:/workspace/outputs:rw",
                self.image,
                "python", "/workspace/workspace/generated.py"
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "success": result.returncode == 0,
        }