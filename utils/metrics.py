import torch

def accuracy(output, y_true):
    y_pred = output.argmax(dim=1)
    score = torch.sum(y_pred == y_true) / torch.numel(y_pred)
    return score