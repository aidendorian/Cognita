import subprocess
import uuid
from tools.filesystem import FileSystem

class PythonSandbox:
    def __init__(self, project_id: int, run_id: str | None = None, timeout: int = 240):
        if project_id <= 0:
            raise ValueError("project_id must be positive")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self.project_id = project_id
        self.run_id = run_id
        self.fs = FileSystem(project_id, run_id=run_id)
        self.image = "research-sandbox"
        self.timeout = timeout

    def run(self, code: str) -> dict:
        if not isinstance(code, str):
            raise TypeError("code must be a string")

        execution_id = uuid.uuid4().hex[:8]
        script_name = f"workspace/run_{execution_id}.py"
        self.fs.write(script_name, code)
        host_path = self.fs.base_path.resolve()

        try:
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--memory",
                    "512m",
                    "--cpus",
                    "1.0",
                    "--pids-limit",
                    "100",
                    "--read-only",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges:true",
                    "--user",
                    "1000:1000",
                    "--tmpfs",
                    "/tmp:rw,noexec,nosuid,size=64m",
                    "-v",
                    f"{host_path}/sources:/workspace/sources:ro",
                    "-v",
                    f"{host_path}/workspace:/workspace/workspace:rw",
                    "-v",
                    f"{host_path}/outputs:/workspace/outputs:rw",
                    self.image,
                    "python",
                    f"/workspace/{script_name}",
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            self._delete_script(script_name)
            return {
                "stdout": "",
                "stderr": f"Execution timed out after {self.timeout}s.",
                "returncode": -1,
                "success": False,
            }
        except (OSError, subprocess.SubprocessError) as exc:
            self._delete_script(script_name)
            return {
                "stdout": "",
                "stderr": f"Sandbox execution failed: {exc}",
                "returncode": -1,
                "success": False,
            }
        finally:
            self._delete_script(script_name)

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "success": result.returncode == 0,
        }

    def _delete_script(self, script_name: str) -> None:
        try:
            if self.fs.exists(script_name):
                self.fs.delete(script_name)
        except Exception:
            pass