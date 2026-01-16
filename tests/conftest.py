import os
from pathlib import Path
import sys

# Ensure project root is on sys.path for tests
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env into os.environ so integration tests can pick up keys
from core.config_adapter import DotEnvConfigSource  # noqa: E402

dotenv = DotEnvConfigSource(path=ROOT / ".env")
dotenv._load()
for key, val in dotenv._cache.items():
    os.environ.setdefault(key, val)
