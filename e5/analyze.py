"""Compatibility entry point; implementation moved to E7.scoring."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from E7.scoring import *
if __name__ == "__main__":
    main()
