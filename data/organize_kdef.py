"""Sort the original KDEF images into one folder per emotion.

KDEF file names encode the picture, e.g. AF01ANS.JPG:
session (A/B), gender (F/M), identity (01-35), expression (AF, AN, DI, HA, NE, SA, SU)
and angle (FL, HL, S, HR, FR). Only front-facing (S) images are kept by default.

Usage: python organize_kdef.py <path to extracted KDEF folder> [--all-angles]
"""

import shutil
import sys
from pathlib import Path

EXPRESSIONS = {
    "AF": "fear",
    "AN": "angry",
    "DI": "disgust",
    "HA": "happy",
    "NE": "neutral",
    "SA": "sad",
    "SU": "surprise",
}

source = Path(sys.argv[1])
all_angles = "--all-angles" in sys.argv
target = Path(__file__).parent / "images" / "kdef"

copied = 0
for path in source.rglob("*"):
    name = path.stem.upper()
    if path.suffix.upper() != ".JPG" or len(name) < 7 or name[4:6] not in EXPRESSIONS:
        continue
    if not all_angles and name[6:] != "S":
        continue
    folder = target / EXPRESSIONS[name[4:6]]
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, folder / path.name)
    copied += 1

print(f"Copied {copied} images into {target}")
