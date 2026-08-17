import subprocess
import uuid
from tools.filesystem import FileSystem

class PythonSandbox:
    def __init__(self, project_id: int, timeout: int = 30):
        self.project_id = project_id
        self.fs = FileSystem(project_id)
        self.image = "research-sandbox"
        self.timeout = timeout

    def run(self, code: str) -> dict:
        run_id = uuid.uuid4().hex[:8]
        script_name = f"workspace/run_{run_id}.py"
        self.fs.write(script_name, code)
        host_path = self.fs.base_path.resolve()

        try:
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--network", "none",
                    "--memory", "512m",
                    "--pids-limit", "100",
                    "--read-only",
                    "--tmpfs", "/tmp",
                    "-v", f"{host_path}/sources:/workspace/sources:ro",
                    "-v", f"{host_path}/workspace:/workspace/workspace:rw",
                    "-v", f"{host_path}/outputs:/workspace/outputs:rw",
                    self.image,
                    "python", f"/workspace/{script_name}",
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            if self.fs.exists(script_name):
                self.fs.delete(script_name)
            return {    
                "stdout": "",
                "stderr": f"Execution timed out after {self.timeout}s.",
                "returncode": -1,
                "success": False,
            }

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "success": result.returncode == 0,
        }