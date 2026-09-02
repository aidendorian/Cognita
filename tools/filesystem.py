from pathlib import Path

MAX_READ_BYTES = 10 * 1024 * 1024
DEFAULT_WRITE_ENCODING = "utf-8"

class FileSystem:
    def __init__(self, project_id: int, run_id: str | None = None):
        if project_id <= 0:
            raise ValueError("project_id must be a positive integer")

        self.project_id = project_id
        self.run_id = run_id

        project_root = Path("data") / "projects" / str(project_id)
        self.project_path = project_root.resolve()

        if run_id is not None:
            if not run_id.strip():
                raise ValueError("run_id cannot be empty")

            if "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
                raise ValueError("Invalid run_id")

            self.base_path = (
                self.project_path / "runs" / run_id
            ).resolve()
        else:
            self.base_path = self.project_path

        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, target: str) -> Path:
        if not isinstance(target, str):
            raise TypeError("target must be a string")

        if not target:
            raise ValueError("Path cannot be empty")

        full_path = (self.base_path / target).resolve()

        if not full_path.is_relative_to(self.base_path):
            raise ValueError(
                f"Path escapes filesystem root: {target!r}"
            )
        return full_path

    def exists(self, filename: str) -> bool:
        try:
            return self._resolve_path(filename).exists()
        except ValueError:
            return False

    def write(self, filename: str, content: str) -> None:
        if not isinstance(content, str):
            raise TypeError("content must be a string")

        path = self._resolve_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open(
            mode="w",
            encoding=DEFAULT_WRITE_ENCODING,
        ) as f:
            f.write(content)

    def read(self, filename: str) -> str:
        path = self._resolve_path(filename)

        if not path.is_file():
            raise FileNotFoundError(filename)

        size = path.stat().st_size
        if size > MAX_READ_BYTES:
            raise ValueError(
                f"{filename} is {size // 1024}KB — "
                f"too large to read into memory "
                f"(limit {MAX_READ_BYTES // 1024}KB)"
            )

        with path.open(
            mode="r",
            encoding=DEFAULT_WRITE_ENCODING,
        ) as f:
            return f.read()

    def write_bytes(self, filename: str, content: bytes) -> None:
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")

        path = self._resolve_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open(mode="wb") as f:
            f.write(content)

    def read_bytes(self, filename: str) -> bytes:
        path = self._resolve_path(filename)

        if not path.is_file():
            raise FileNotFoundError(filename)

        size = path.stat().st_size
        if size > MAX_READ_BYTES:
            raise ValueError(
                f"{filename} is {size // 1024}KB — "
                f"too large to read into memory "
                f"(limit {MAX_READ_BYTES // 1024}KB)"
            )

        with path.open(mode="rb") as f:
            return f.read()

    def list_files(self, folder: str = "") -> list[str]:
        folder_path = (
            self._resolve_path(folder)
            if folder
            else self.base_path
        )

        if not folder_path.exists():
            return []

        if not folder_path.is_dir():
            raise NotADirectoryError(folder)

        return [
            str(path.relative_to(self.base_path))
            for path in folder_path.iterdir()
            if path.is_file()
        ]

    def delete(self, filename: str) -> None:
        path = self._resolve_path(filename)

        if path.exists():
            if not path.is_file():
                raise IsADirectoryError(filename)

            path.unlink()