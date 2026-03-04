import yaml
from pathlib import Path

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

CFG = load_config()