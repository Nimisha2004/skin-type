import os
from pathlib import Path

root = Path("clean_dataset")

for split in ["train", "valid", "test"]:
    print(f"\n--- {split.upper()} ---")
    for cls in ["oily", "normal", "dry"]:
        folder = root / split / cls
        if folder.exists():
            count = len(list(folder.glob("*")))
            print(f"{cls}: {count}")
        else:
            print(f"{cls}: 0")
