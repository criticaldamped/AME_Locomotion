"""Use only this source tree; no installation or environment mutation."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'rsl_rl'))
sys.path.insert(0, str(ROOT / 'source/ame_locomotion'))
