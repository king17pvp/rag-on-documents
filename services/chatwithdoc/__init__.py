import os
import sys

# Add all lib source directories to Python path so that lib packages
# (base, llm, database, etc.) are importable by their short package names.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _lib in [
    "base",
    "database",
    "document_parser",
    "embedding",
    "llm",
    "open_search",
    "storage_handler",
]:
    _src = os.path.join(_ROOT, "libs", _lib, "src")
    if os.path.exists(_src) and _src not in sys.path:
        sys.path.insert(0, _src)
