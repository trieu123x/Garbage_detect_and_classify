import os
import shutil
import random
from pathlib import Path

# Paths
base_dir = Path("c:/Users/admin/Downloads/Phan_loai_rac(12cls)")
folders_to_merge = ["original"]
output_dir = base_dir / "dataset"
train_dir = output_dir / "train"
val_dir = output_dir / "val"
test_dir = output_dir / "test"

# Delete existing dataset directory if it exists to ensure a clean split
if output_dir.exists():
    print(f"Removing existing dataset directory: {output_dir}")
    shutil.rmtree(output_dir)

classes = [d for d in os.listdir(base_dir / "original") if os.path.isdir(base_dir / "original" / d)]

# Create output directories
for cls in classes:
    os.makedirs(train_dir / cls, exist_ok=True)
    os.makedirs(val_dir / cls, exist_ok=True)
    os.makedirs(test_dir / cls, exist_ok=True)

for cls in classes:
    all_images = []
    for folder in folders_to_merge:
        folder_path = base_dir / folder / cls
        if folder_path.exists():
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append({
                        "src_path": folder_path / img_name,
                        "folder_name": folder,
                        "img_name": img_name
                    })
    
    random.seed(42)
    random.shuffle(all_images)
    
    # 70% train, 15% val, 15% test
    n = len(all_images)
    train_idx = int(n * 0.7)
    val_idx = int(n * 0.85) # 0.7 + 0.15
    
    train_images = all_images[:train_idx]
    val_images = all_images[train_idx:val_idx]
    test_images = all_images[val_idx:]
    
    for img_info in train_images:
        dest_name = f"{img_info['folder_name']}_{img_info['img_name']}"
        shutil.copy(img_info['src_path'], train_dir / cls / dest_name)
        
    for img_info in val_images:
        dest_name = f"{img_info['folder_name']}_{img_info['img_name']}"
        shutil.copy(img_info['src_path'], val_dir / cls / dest_name)

    for img_info in test_images:
        dest_name = f"{img_info['folder_name']}_{img_info['img_name']}"
        shutil.copy(img_info['src_path'], test_dir / cls / dest_name)
        
    print(f"Class '{cls}': {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

print("Dataset successfully created and split into train/val/test (70/15/15)!")
