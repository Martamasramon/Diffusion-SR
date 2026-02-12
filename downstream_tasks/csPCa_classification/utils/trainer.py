import os
from sched import scheduler
import torch
from tqdm import tqdm
from utils.metrics import accuracy, class_accuracy
from utils.test_validate import val_test_model
from sklearn.metrics import roc_curve, auc

def train_single_epoch(model, dataloader, criterion, optimizer, device):

    train_loss = 0.0
    all_preds = []
    all_targets = []

    model.train()

    for images, lesion_mask, metadata, labels in tqdm(dataloader, desc='Training', unit='batch'):

        images = images.to(device)
        lesion_mask = lesion_mask.to(device)
        metadata = metadata.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        train_logits = model(images, lesion_mask, metadata)
        loss = criterion(train_logits, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_probs = torch.sigmoid(train_logits).detach().cpu().numpy()

        all_preds.extend(train_probs.tolist())
        all_targets.extend(labels.detach().cpu().numpy().tolist())
    
    epoch_loss = train_loss / len(dataloader.dataset)

    train_accuracy = accuracy(torch.tensor(all_targets), torch.tensor(all_preds))
    train_class_accuracy = class_accuracy(torch.tensor(all_targets), torch.tensor(all_preds))

    print(f'Epoch Train Loss: {epoch_loss}, Epoch Train Accuracy: {train_accuracy}, Epoch Train Class Accuracy: {train_class_accuracy}')

    return all_targets, all_preds, epoch_loss, train_accuracy, train_class_accuracy


def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs, output_dir, timestamp, scheduler=None):
    
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    train_class_accuracies, val_class_accuracies = {}, {}
    best_val_loss = float('inf')
    best_epoch = -1
    best_train_labels, best_train_preds, best_val_labels, best_val_preds = None, None, None, None

    best_checkpoint_dir = f"{output_dir}/saved_models"
    if not os.path.exists(best_checkpoint_dir):
        os.makedirs(best_checkpoint_dir)

    for epoch in range(1, num_epochs+1):

        print(f'\nEpoch {epoch}/{num_epochs}')

        train_epoch_labels, train_epoch_preds, train_epoch_loss, train_epoch_accuracy, train_epoch_class_accuracy = train_single_epoch(model, train_dataloader, criterion, optimizer, device)
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        train_class_accuracies[epoch-1] = train_epoch_class_accuracy
    

        val_epoch_labels, val_epoch_preds, val_epoch_loss, val_epoch_accuracy, val_epoch_class_accuracy = val_test_model('Val', model, val_dataloader, criterion, device)
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)
        val_class_accuracies[epoch-1] = val_epoch_class_accuracy

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_epoch = epoch
            best_train_labels, best_train_preds, best_val_labels, best_val_preds = train_epoch_labels, train_epoch_preds, val_epoch_labels, val_epoch_preds
            best_checkpoint_path = f'{best_checkpoint_dir}/best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, best_checkpoint_path)
            print('Best model saved.')

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                fpr, tpr, _ = roc_curve(val_epoch_labels, val_epoch_preds)
                val_roc_auc = auc(fpr, tpr)
                scheduler.step(val_roc_auc)
            else:
                scheduler.step()

    print('Best epoch:', best_epoch, 'Validation Loss at this epoch:', best_val_loss, 'Train Loss at this epoch:', train_losses[best_epoch-1])

    return best_train_labels, best_train_preds, best_val_labels, best_val_preds, train_losses, val_losses, train_accuracies, val_accuracies, train_class_accuracies, val_class_accuracies, best_checkpoint_path