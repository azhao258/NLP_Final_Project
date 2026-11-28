"""Compatibility entry point; implementation moved to E6.analyze_test_suite."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from E6.analyze_test_suite import *
if __name__ == "__main__":
    main()
