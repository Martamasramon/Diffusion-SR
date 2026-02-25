import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import random
from torchinfo import summary
from torch.utils.data import DataLoader
from utils.dataset import PicaiDataset
from utils.trainer import train_model
from utils.test_validate import val_test_model
from utils.visualize import plot_training_curves, plot_classification_metrics
from utils.preprocess_metadata import preprocess_metadata
from cs_classification_model import MultimodalPICAINet

def set_random_seed(seed):
    np.random.seed(seed) 
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(output_dir, timestamp, debug=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 60
    batch_size = 32
    random_seed = 13

    metadata_csv_path = '/cluster/project7/backup_masramon/picai_marksheet.csv'
    # metadata_csv_path = './data/picai_marksheet.csv'
    target_size = (64, 64)

    set_random_seed(random_seed)

    print("Preprocessing the metadata...")
    train_metadata_X_df, train_metadata_y, val_metadata_X_df, val_metadata_y, test_metadata_X_df, test_metadata_y = preprocess_metadata(metadata_csv_path, random_state=random_seed)

    # Make the labels balanced by downsampling the majority class in the training set
    balance_training_data = False
    if balance_training_data:
        pos_idx = train_metadata_y[train_metadata_y == 1].index
        neg_idx = train_metadata_y[train_metadata_y == 0].index
        neg_sampled_idx = np.random.choice(neg_idx, size=len(pos_idx), replace=False)
        balanced_idx = np.concatenate([pos_idx, neg_sampled_idx])
        np.random.shuffle(balanced_idx)
        train_metadata_X_df = train_metadata_X_df.loc[balanced_idx]
        train_metadata_y = train_metadata_y.loc[balanced_idx]

    if debug:
        print("Debug mode enabled: using smaller dataset and fewer epochs and a smaller batch size")
        batch_size = 4
        num_epochs = 5
        train_metadata_X_df, train_metadata_y, val_metadata_X_df, val_metadata_y, test_metadata_X_df, test_metadata_y = \
            train_metadata_X_df[:32], train_metadata_y[:32], val_metadata_X_df[:16], val_metadata_y[:16], test_metadata_X_df[:16], test_metadata_y[:16] 
        print(f"Debug Train metadata shape: {train_metadata_X_df.shape}, Debug Val metadata shape: {val_metadata_X_df.shape}, Debug Test metadata shape: {test_metadata_X_df.shape}")

    print("Creating Datasets and DataLoaders...")
    
    train_dataset = PicaiDataset(
        metadata_X_df=train_metadata_X_df,
        img_dir='./data/',
        labels=train_metadata_y.tolist(),
        target_size=target_size
    )

    val_dataset = PicaiDataset(
        metadata_X_df=val_metadata_X_df,
        img_dir='./data/',
        labels=val_metadata_y.tolist(),
        target_size=target_size
    )

    test_dataset = PicaiDataset(
        metadata_X_df=test_metadata_X_df,
        img_dir='./data/',
        labels=test_metadata_y.tolist(),
        target_size=target_size
    )

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )

    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    print(f"DataLoaders created with batch size {batch_size}.")

    cs_model = MultimodalPICAINet(metadata_input_dim=train_metadata_X_df.shape[1] - 2).to(device)

    pos_weight = torch.tensor([ (len(train_metadata_y) - sum(train_metadata_y)) / sum(train_metadata_y) ]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight/2)

    weight_decay = 1e-3
    optimizer = torch.optim.AdamW(cs_model.parameters(), lr=1e-3, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print("Model Summary: ")
    summary(cs_model, input_size=[(1, 3, 64, 64), (1, 1, 64, 64), (1, train_metadata_X_df.shape[1] - 2)], device=device.type)

    for name, param in cs_model.named_parameters():
        if param.requires_grad:
            print(name)

    print("Starting training...")

    best_train_labels, best_train_preds, best_val_labels, best_val_preds, \
        train_losses, val_losses, train_accuracies, val_accuracies, train_class_accuracies, val_class_accuracies, \
             best_checkpoint_path = train_model(
        model=cs_model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        output_dir=output_dir, 
        timestamp=timestamp,
        scheduler=scheduler
    )

    print("Training complete. Plotting training curves and classification metrics...")

    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, train_class_accuracies, val_class_accuracies, output_dir, timestamp)

    train_f1_score = plot_classification_metrics('train', np.array(best_train_labels), np.array(best_train_preds), output_dir, timestamp)
    print(f'F1 Score on Training Set: {train_f1_score}')

    val_f1_score = plot_classification_metrics('val', np.array(best_val_labels), np.array(best_val_preds), output_dir, timestamp)
    print(f'F1 Score on Validation Set: {val_f1_score}')

    print("Loading best model checkpoint for final evaluation on test set...")
    cs_model.load_state_dict(torch.load(best_checkpoint_path, weights_only=True)['model_state_dict'])

    print("Evaluating on test set...")
    test_labels, test_preds, test_loss, test_accuracy, test_class_accuracy = val_test_model('Test', cs_model, test_loader, criterion, device)
    
    test_f1_score = plot_classification_metrics('test', np.array(test_labels), np.array(test_preds), output_dir, timestamp)
    print(f'F1 Score on Test Set: {test_f1_score}')


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("Error: Please provide the output directory and timestamp as command-line arguments. Example usage: python train.py outputs/ 20240601_123456")
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('output_dir', type=str, help='Directory to save outputs like model checkpoints and plots') 
        parser.add_argument('timestamp', type=str, help='Timestamp string to uniquely identify this training run (e.g., 20240601_123456)')
        parser.add_argument('--debug', action='store_true', help='Run in debug mode with smaller dataset and fewer epochs and a smaller batch size')
        args = parser.parse_args()
        debug = args.debug
        train(output_dir=args.output_dir, timestamp=args.timestamp, debug=debug)