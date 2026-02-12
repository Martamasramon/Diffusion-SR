import torch

def accuracy(y_true, y_pred):
    y_pred_labels = (y_pred >= 0.5).long()
    correct = (y_true == y_pred_labels).float()
    accuracy = correct.sum() / len(correct)
    return accuracy.item() * 100

def class_accuracy(y_true, y_pred):
    class_accuracy = {}
    y_pred_labels = (y_pred >= 0.5).long()
    
    for cls_label in torch.unique(y_true):
        cls_mask = (y_true == cls_label)
        correct = (y_true[cls_mask] == y_pred_labels[cls_mask]).float()
        accuracy = correct.sum() / len(correct)
        class_accuracy[int(cls_label.item())] = accuracy.item() * 100
    return class_accuracy