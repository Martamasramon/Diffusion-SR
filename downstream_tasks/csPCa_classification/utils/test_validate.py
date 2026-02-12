import torch
from tqdm import tqdm
from utils.metrics import accuracy, class_accuracy

def val_test_model(split_type, model, dataloader, criterion, device):

    val_test_loss = 0.0
    all_preds = []
    all_targets = []

    model.eval()

    with torch.no_grad():
        for images, lesion_mask, metadata, labels in tqdm(dataloader, desc=f'{split_type.title()} Evaluation', unit='batch'):
            images = images.to(device)
            lesion_mask = lesion_mask.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)

            val_test_logits = model(images, lesion_mask, metadata)
            loss = criterion(val_test_logits, labels)

            val_test_loss += loss.item() * images.size(0)
            val_test_probs = torch.sigmoid(val_test_logits).detach().cpu().numpy()

            all_preds.extend(val_test_probs.tolist())
            all_targets.extend(labels.detach().cpu().numpy().tolist())
    
    epoch_loss = val_test_loss / len(dataloader.dataset)

    val_accuracy = accuracy(torch.tensor(all_targets), torch.tensor(all_preds))
    val_class_accuracy = class_accuracy(torch.tensor(all_targets), torch.tensor(all_preds))

    print(f'Epoch {split_type.title()} Loss: {epoch_loss}, Epoch {split_type.title()} Accuracy: {val_accuracy}, Epoch {split_type.title()} Class Accuracy: {val_class_accuracy}')

    return all_targets, all_preds, epoch_loss, val_accuracy, val_class_accuracy