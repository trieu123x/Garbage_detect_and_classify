import os
from pathlib import Path

def count_dataset():
    base_dir = Path("dataset")
    if not base_dir.exists():
        print("Dataset directory not found.")
        return

    for split in ["train", "val", "test"]:
        split_dir = base_dir / split
        if not split_dir.exists():
            continue
        print(f"\n--- {split.upper()} ---")
        classes = sorted([d for d in os.listdir(split_dir) if (split_dir / d).is_dir()])
        total = 0
        for cls in classes:
            cls_dir = split_dir / cls
            count = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"{cls:15}: {count}")
            total += count
        print(f"{'TOTAL':15}: {total}")

if __name__ == '__main__':
    count_dataset()
