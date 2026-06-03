"""
Root-level conftest.py — two jobs:
1. Its presence tells pytest to use the repo root as rootdir (so sys.path
   gets the repo root prepended automatically, making `import src.*` work).
2. Belt-and-suspenders: explicitly inserts the repo root at the front of
   sys.path so `from src.config import …` always resolves, regardless of
   how pytest was invoked or which import mode is active.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
