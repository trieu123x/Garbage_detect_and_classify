import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from pathlib import Path

# Thêm thư mục training vào sys.path để import cùng cấp
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import get_data_loaders
from train import train_model
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from evaluate import evaluate_model, plot_confusion_matrix

def main():
    base_dir = Path(__file__).resolve().parent.parent  # root project dir
    data_dir = base_dir / "dataset"

    if not os.path.exists(data_dir):
        print("Dataset directory not found. Please run prepare_data.py first.")
        return

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device} | GPUs available: {num_gpus}")

    # Data
    dataloaders, dataset_sizes, class_names = get_data_loaders(data_dir, batch_size=16)
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # Model (UPDATED)
    model_ft = resnet50(weights=ResNet50_Weights.DEFAULT)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    # Freeze backbone (IMPORTANT)
    for param in model_ft.parameters():
        param.requires_grad = False

    for name, param in model_ft.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True

    # Use DataParallel if multiple GPUs are available
    if num_gpus > 1:
        print(f"Using DataParallel on {num_gpus} GPUs")
        model_ft = nn.DataParallel(model_ft)

    model_ft = model_ft.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer (only train fc)
    optimizer_ft = optim.AdamW(
        filter(lambda p: p.requires_grad, model_ft.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )
    
    # Train
    num_epochs = 50
    best_model = train_model(
        model_ft,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer_ft,
        device,
        num_epochs=num_epochs,
 
    )

    # Save model (unwrap DataParallel if needed)
    save_path = base_dir / "best_model.pth"
    state_dict = best_model.module.state_dict() if isinstance(best_model, nn.DataParallel) else best_model.state_dict()
    torch.save(state_dict, save_path)
    print(f"Model successfully saved to {save_path}")


    # Load best model for testing
    print("\nLoading best model for testing...")
    model_ft.load_state_dict(state_dict)
    
    # Evaluate on test set
    y_true, y_pred = evaluate_model(model_ft, dataloaders['test'], device)
    
    # Print Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Plot Confusion Matrix
    cm_path = base_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)

if __name__ == '__main__':
    main()