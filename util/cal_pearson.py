# PYTORCH version of the vlaai original code.
import torch
import pdb

def pearson_correlation(y_true, y_pred, axis=1):

    y_true_mean = torch.mean(y_true, dim=axis, keepdim=True)
    y_pred_mean = torch.mean(y_pred, dim=axis, keepdim=True)

    # Compute the numerator and denominator of the pearson correlation.
    numerator = torch.sum((y_true - y_true_mean) * (y_pred - y_pred_mean),
        dim=axis,
        keepdim=False)

    std_true = torch.sum((y_true - y_true_mean)**2, dim=axis, keepdim=False)
    std_pred = torch.sum((y_pred - y_pred_mean)**2, dim=axis, keepdim=False)
    denominator = torch.sqrt(std_true * std_pred)
    
    pearsonR = torch.div(numerator, denominator + 1e-6)

    assert torch.all(torch.lt(pearsonR, 1)) and torch.all(torch.gt(pearsonR, -1)), "Loss contains values outside the range of -1 to 1"

    return pearsonR


def pearson_loss(y_true, y_pred, axis=1):
    return -pearson_correlation(y_true, y_pred, axis=axis)

def pearson_metric(y_true, y_pred, axis=1):
    return pearson_correlation(y_true, y_pred, axis=axis)
    
def l1_loss(y_true, y_pred, axis=1):
    l1_dist = torch.abs(y_true - y_pred)
    l1_loss = torch.mean(l1_dist, axis = axis, keepdim=False)
    return l1_loss