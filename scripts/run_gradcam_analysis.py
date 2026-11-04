"""Convenience entry point; run from the repository root."""
import sys
from pathlib import Path

# Support ``python scripts/run_gradcam_analysis.py`` as well as module execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain import main


if __name__ == "__main__":
    main()

