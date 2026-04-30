"""
将 test.txt 中列出的文件复制到 bamboo/test/ 目录。
用法：python tools/collect_test.py
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
TEST_TXT = ROOT / "test.txt"
DST_DIR = ROOT / "bamboo" / "test"

DST_DIR.mkdir(parents=True, exist_ok=True)

missing = []
copied = 0

with open(TEST_TXT) as f:
    for line in f:
        rel = line.strip()
        if not rel:
            continue
        src = ROOT / rel
        if not src.exists():
            missing.append(rel)
            continue
        shutil.copy2(src, DST_DIR / src.name)
        copied += 1

print(f"Copied {copied} file(s) to {DST_DIR}")
if missing:
    print(f"Missing ({len(missing)}):")
    for p in missing:
        print(f"  {p}")
