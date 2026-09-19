from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)

class DirectoryLoader:
    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.supported_extensions = {".md", ".txt"}

    def load(self) -> List[Document]:
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory.resolve()}")
        if not self.directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.directory.resolve()}")

        documents = []
        for file_path in self.directory.rglob("*"):
            if not file_path.is_file():
                continue
            
            if file_path.suffix.lower() not in self.supported_extensions:
                # We skip silently unless explicitly supplied, but here we just glob
                continue
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                if not text.strip():
                    continue  # Ignore empty files
                
                doc = Document(
                    text=text,
                    metadata={"source": file_path.name}
                )
                documents.append(doc)
            except UnicodeDecodeError:
                raise ValueError(f"File {file_path.name} is not valid UTF-8.")
            except Exception as e:
                raise IOError(f"Error reading file {file_path.name}: {e}")
                
        return documents
