import os
from pathlib import Path


def runtime_dir(env_name: str, default_name: str) -> Path:
    configured = os.getenv(env_name, "").strip()
    if configured:
        path = Path(configured)
    elif os.getenv("VERCEL"):
        path = Path("/tmp") / default_name
    else:
        path = Path(f"./{default_name}")
    path.mkdir(parents=True, exist_ok=True)
    return path
