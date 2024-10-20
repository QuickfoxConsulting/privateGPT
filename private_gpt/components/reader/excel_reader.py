import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

class ExcelReader(BaseReader):
    def load_data(self, file_path: Path, extra_info: Optional[Dict] = None) -> List[Document]:
        if extra_info is not None:
            if not isinstance(extra_info, dict):
                raise TypeError("extra_info must be a dictionary.")
        data = pd.read_excel(file_path).to_json(orient='records')
        return [Document(text=data, metadata=extra_info)]