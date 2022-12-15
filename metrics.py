import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

# PA : TP+TN/TP+TN+FP+FN
def pixel_accuracy(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5

    _output_ = output <= 0.5
    _target_ = target <= 0.5

    TP = (output_ & target_).sum()
    TN = (_output_ & _target_).sum()
    FP = (output_ & _target_).sum()
    FN = (_output_ & target_).sum()


    return (TP + TN + smooth) / (TP + TN + FP + FN +smooth)

# SE : TP / TP + FN
def sensitivity(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5

    _output_ = output <= 0.5
    _target_ = target <= 0.5

    TP = (output_ & target_).sum()
    TN = (_output_ & _target_).sum()
    FP = (output_ & _target_).sum()
    FN = (_output_ & target_).sum()

    return (TP + smooth) / (TP + FN + smooth)

def specificity(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5

    _output_ = output <= 0.5
    _target_ = target <= 0.5

    TP = (output_ & target_).sum()
    TN = (_output_ & _target_).sum()
    FP = (output_ & _target_).sum()
    FN = (_output_ & target_).sum()

    return (TN + smooth) / (TN + FP + smooth)

def precision(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5

    _output_ = output <= 0.5
    _target_ = target <= 0.5

    TP = (output_ & target_).sum()

    TN = (_output_ & _target_).sum()
    FP = (output_ & _target_).sum()
    FN = (_output_ & target_).sum()

    return (TP + smooth) / (TP + FP + smooth)

def F1_Score(output, target):
    return 2 * ((precision(output, target) * sensitivity(output, target)) /
                (precision(output, target) + sensitivity(output, target)))