import os
import sys
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import get_data_loaders

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    print("\nRunning evaluation on test set...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def main():
    # Dynamic paths
    base_dir = Path(__file__).resolve().parent.parent  # root project dir
    data_dir = base_dir / "dataset"
    model_path = base_dir / "weights" / "classifier" / "best_resnet50_garbage.pth"
    
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data for class names and test loader
    dataloaders, dataset_sizes, class_names = get_data_loaders(data_dir, batch_size=16)
    num_classes = len(class_names)
    
    # Initialize model
    model = resnet50(weights=None) # We load custom weights
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load state dict
    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    # Evaluate
    y_true, y_pred = evaluate_model(model, dataloaders['test'], device)
    
    # Print Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Plot
    cm_path = base_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)

if __name__ == '__main__':
    main()
