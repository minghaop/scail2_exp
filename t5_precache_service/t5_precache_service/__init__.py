"""T5 prompt precache service package within the SCAIL-2 source tree."""

from pathlib import Path
import sys


# This subproject reuses the canonical cache contract and model implementation
# from its parent SCAIL-2 source tree.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
