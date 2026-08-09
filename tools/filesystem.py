from pathlib import Path

class FileSystem:
    def __init__(self, project_id: int):
        self.base_path = Path(f"data/projects/{project_id}")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def _resolve_path(self, target: str):
        full_path = (self.base_path/target).resolve()
        if not full_path.is_relative_to(self.base_path.resolve()):
            raise ValueError(f"Path escapes project root: {target}")
        return full_path
    
    def write(self, filename, content):
        path = self._resolve_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode="w", encoding="utf-8") as file:
            file.write(content)
                
    def writeBytes(self, filename, content):
        path = self._resolve_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode="wb") as file:
            file.write(content)
            
    def read(self, filename):
        path = self._resolve_path(filename)
        with open(path, mode="r", encoding="utf-8") as file:
            return file.read()
                    
    def readBytes(self, filename):
        path = self._resolve_path(filename)
        with open(path, mode="rb") as file:
            return file.read()
        
    def list_files(self, folder):
        folder_path = self._resolve_path(folder)
        return [str(x.relative_to(self.base_path)) for x in folder_path.iterdir() if x.is_file()]