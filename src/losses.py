"""Loss functions for multi-label glioma MRI segmentation."""

import torch
from torch import nn
from monai.losses import DiceLoss


class MultiLabelDiceBCELoss(nn.Module):
    """
    Combine sigmoid Dice loss and binary cross-entropy.

    Intended for overlapping binary segmentation regions,
    such as WT, TC, ET and RC.

    Inputs:
        logits: Tensor of shape (B, C, H, W, D).
        targets: Binary tensor with the same shape.

    The input must contain raw logits, not probabilities.
    """

    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super().__init__()

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        self.dice_loss = DiceLoss(
            include_background=True,
            to_onehot_y=False,
            sigmoid=True,
            softmax=False,
            squared_pred=False,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
            reduction="mean",
        )

        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction="mean"
        )

    def forward(self, logits, targets):
        """Return the weighted Dice + BCE loss."""

        if logits.shape != targets.shape:
            raise ValueError(
                "Logits and targets must have matching shapes. "
                f"Received {logits.shape} and {targets.shape}."
            )

        targets = targets.to(dtype=logits.dtype)

        dice = self.dice_loss(logits, targets)
        bce = self.bce_loss(logits, targets)

        return (
            self.dice_weight * dice
            + self.bce_weight * bce
        )
