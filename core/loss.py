import torch
import torch.nn as nn
import torch.nn.functional as F

class RecallFocusedComboLoss(nn.Module):
    def __init__(self, pos_weight_val=5.0, tversky_alpha=0.3, tversky_beta=0.7):
        super().__init__()
        self.register_buffer('pos_weight', torch.tensor([pos_weight_val], dtype=torch.float32))
        self.alpha = tversky_alpha
        self.beta = tversky_beta

    def forward(self, preds, masks):
        bce_loss = F.binary_cross_entropy_with_logits(
            preds,
            masks,
            pos_weight=self.pos_weight
        )

        probs = torch.sigmoid(preds)
        probs_flat = probs.view(-1)
        masks_flat = masks.view(-1)

        tp = (probs_flat * masks_flat).sum()
        fp = (probs_flat * (1 - masks_flat)).sum()
        fn = ((1 - probs_flat) * masks_flat).sum()

        tversky_coef = (tp + 1e-6) / (tp + self.alpha * fp + self.beta * fn + 1e-6)
        tversky_loss = 1.0 - tversky_coef

        return bce_loss + tversky_loss