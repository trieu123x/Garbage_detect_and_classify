import torch
import copy
from torch.optim import lr_scheduler
from torch import nn

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs):
    # Decay LR by 0.1 every 5 epochs
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )
    

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("Starting Training...")

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()

                if (batch_idx + 1) % 20 == 0:
                    print(f'  [{phase}] Batch {batch_idx+1}/{len(dataloaders[phase])} | '
                          f'Loss: {loss.item():.4f}')

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f'>> {phase} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'  [*] New best val acc: {best_acc:.4f} — model saved')

    print(f'\nTraining complete. Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model