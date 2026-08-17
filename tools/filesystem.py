from pathlib import Path

MAX_READ_BYTES = 10 * 1024 * 1024

class FileSystem:
    def __init__(self, project_id: int):
        self.base_path = Path(f"data/projects/{project_id}")
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, target: str) -> Path:
        full_path = (self.base_path / target).resolve()
        if not full_path.is_relative_to(self.base_path.resolve()):
            raise ValueError(f"Path escapes project root: {target!r}")
        return full_path

    def exists(self, filename: str) -> bool:
        try:
            return self._resolve_path(filename).exists()
        except ValueError:
            return False

    def write(self, filename: str, content: str) -> None:
        path = self._resolve_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode="w", encoding="utf-8") as f:
            f.write(content)

    def read(self, filename: str) -> str:
        path = self._resolve_path(filename)
        if path.stat().st_size > MAX_READ_BYTES:
            raise ValueError(
                f"{filename} is {path.stat().st_size // 1024}KB — "
                f"too large to read into memory (limit {MAX_READ_BYTES // 1024}KB)"
            )
        with open(path, mode="r", encoding="utf-8") as f:
            return f.read()

    def write_bytes(self, filename: str, content: bytes) -> None:
        path = self._resolve_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode="wb") as f:
            f.write(content)

    def read_bytes(self, filename: str) -> bytes:
        path = self._resolve_path(filename)
        if path.stat().st_size > MAX_READ_BYTES:
            raise ValueError(
                f"{filename} is {path.stat().st_size // 1024}KB — too large to read"
            )
        with open(path, mode="rb") as f:
            return f.read()

    def list_files(self, folder: str = "") -> list[str]:
        folder_path = self._resolve_path(folder) if folder else self.base_path
        if not folder_path.exists():
            return []
        return [
            str(x.relative_to(self.base_path))
            for x in folder_path.iterdir()
            if x.is_file()
        ]

    def delete(self, filename: str) -> None:
        path = self._resolve_path(filename)
        if path.exists():
            path.unlink()