import os
from pathlib import Path

def check_dataset(dataset_dir="dataset"):
    base_dir = Path(dataset_dir)
    
    if not base_dir.exists():
        print(f"Dataset directory '{dataset_dir}' not found!")
        return

    splits = ["train", "val", "test"]
    
    for split in splits:
        split_dir = base_dir / split
        if not split_dir.exists():
            print(f"Split directory '{split}' not found in '{dataset_dir}'!")
            continue

        print(f"\n--- Checking '{split}' split ---")
        classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(split_dir / d)])
        
        total_images = 0
        for cls in classes:
            cls_dir = split_dir / cls
            images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            count = len(images)
            total_images += count
            print(f"Class '{cls}': {count} images")
            
        print(f"Total classes in {split}: {len(classes)}")
        print(f"Total images in {split}: {total_images}")

if __name__ == "__main__":
    check_dataset()
