import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, confusion_matrix

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, train_class_accuracies, val_class_accuracies, output_dir, timestamp):
     
    epochs = range(1, len(train_losses) + 1)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    axs[0].plot(epochs, train_losses, label='Train Loss')
    axs[0].plot(epochs, val_losses, label='Validation Loss')
    axs[0].set_title('Loss Curves')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(epochs, train_accuracies, label='Train Accuracy')
    axs[1].plot(epochs, val_accuracies, label='Validation Accuracy')
    axs[1].set_title('Accuracy Curves')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(epochs, [train_class_accuracies[epoch-1][0] for epoch in epochs], label='Train Class 0 Accuracy')
    axs[2].plot(epochs, [val_class_accuracies[epoch-1][0] for epoch in epochs], label='Validation Class 0 Accuracy')
    axs[2].set_title('Accuracy Curves: Clinically non-significant tissues')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Accuracy')
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(epochs, [train_class_accuracies[epoch-1][1] for epoch in epochs], label='Train Class 1 Accuracy')
    axs[3].plot(epochs, [val_class_accuracies[epoch-1][1] for epoch in epochs], label='Validation Class 1 Accuracy')
    axs[3].set_title('Accuracy Curves: Clinically significant lesions')
    axs[3].set_xlabel('Epochs')
    axs[3].set_ylabel('Accuracy')
    axs[3].legend()
    axs[3].grid(True)

    # plt.tight_layout()
    # plt.show()

    fig.savefig(f"{output_dir}/training_curves_{timestamp}.png", dpi=300, bbox_inches='tight')

def plot_classification_metrics(split_type, labels, preds, output_dir, timestamp):
    
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(labels, preds)
    pr_auc = auc(recall, precision)

    pred_labels = (preds >= 0.5).astype(int)
    f1_score_value = f1_score(labels, pred_labels)
    cm = confusion_matrix(labels, pred_labels)

    split_type_formatted = {
        'train': 'Training', 'val': 'Validation', 'test': 'Test'
    }

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    axs[0].plot([0, 1], [0, 1], color='red', linestyle='--')
    axs[0].set_title('Receiver Operating Characteristic (ROC) Curve')
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(recall, precision, color='green', label=f'PR curve (AUC = {pr_auc:.2f})')
    axs[1].set_title('Precision-Recall (PR) Curve')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].legend()
    axs[1].grid(True)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[2])
    axs[2].set_title(f'Confusion Matrix: {split_type_formatted[split_type]}')
    axs[2].set_xlabel('Predicted Label')
    axs[2].set_ylabel('True Label')
    # axs[2].grid(True)

    # plt.tight_layout()
    # plt.show()

    fig.savefig(f"{output_dir}/classification_metrics_{split_type}_{timestamp}.png", dpi=300, bbox_inches='tight')

    return f1_score_value