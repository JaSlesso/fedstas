import torch
from torch.utils.data import DataLoader
import numpy as np

def evaluate_model(model, dataset, batch_size=64, device="cpu", num_classes=10):
    """
    Evaluate model and compute accuracy, loss, and Macro-F1.
    
    Returns:
        accuracy (float): Overall accuracy
        avg_loss (float): Average loss
        macro_f1 (float): Macro-averaged F1 score
    """
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct, total = 0, 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    # Per-class metrics for Macro-F1
    true_positives = np.zeros(num_classes)
    false_positives = np.zeros(num_classes)
    false_negatives = np.zeros(num_classes)
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            # Compute per-class metrics
            y_np = y.cpu().numpy()
            preds_np = preds.cpu().numpy()
            
            for c in range(num_classes):
                true_positives[c] += ((preds_np == c) & (y_np == c)).sum()
                false_positives[c] += ((preds_np == c) & (y_np != c)).sum()
                false_negatives[c] += ((preds_np != c) & (y_np == c)).sum()

    avg_loss = total_loss / total
    accuracy = correct / total
    
    # Compute per-class F1 scores
    f1_scores = []
    for c in range(num_classes):
        precision = true_positives[c] / (true_positives[c] + false_positives[c] + 1e-10)
        recall = true_positives[c] / (true_positives[c] + false_negatives[c] + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        f1_scores.append(f1)
    
    # Macro-F1: average of per-class F1 scores
    macro_f1 = np.mean(f1_scores)
    
    return accuracy, avg_loss, macro_f1
