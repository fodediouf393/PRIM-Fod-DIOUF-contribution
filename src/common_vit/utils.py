from pathlib import Path
import yaml

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)
