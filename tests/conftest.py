"""Make the repo root importable so `import earnings_dpi_study` works when
pytest is invoked from anywhere."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
