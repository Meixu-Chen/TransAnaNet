# PyTorch
import torch

'''
This loss was introduced by Taghanaki et al in their paper "Combo loss: Handling input and output imbalance in 
multi-organ segmentation", retrievable here: https://arxiv.org/abs/1805.02798. Combo loss is a combination of Dice Loss 
and a modified Cross-Entropy function that, like Tversky loss, has additional constants which penalise either false 
positives or false negatives more respectively.
'''

class ComboLoss(torch.nn.Module):
    def __init__(self, args):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        ALPHA = 0.8  # < 0.5 penalises FP more, > 0.5 penalises FN more
        CE_RATIO = 0.5  # weighted contribution of modified CE loss compared to Dice loss
        e = 1e-7

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        inputs = torch.clamp(inputs, e, 1.0 - e)
        out = - (ALPHA * ((targets * torch.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)

        return combo