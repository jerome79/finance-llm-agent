import os, sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print("CWD:", os.getcwd())
print("This file:", __file__)
print("sys.path[0]:", sys.path[0])
print("Top of sys.path (first 5):", sys.path[:5])

project_root = Path(__file__).resolve().parents[1]
print("Expected project root:", project_root)

spec = importlib.util.find_spec("app")
print("find_spec('app'):", spec)
if spec is None:
    sys.path.insert(0, str(project_root))
    print("Added project root to sys.path")
    print("Retry:", importlib.util.find_spec("app"))
