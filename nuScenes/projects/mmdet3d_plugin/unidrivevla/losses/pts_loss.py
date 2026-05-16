"""PtsL1Loss — L1 loss for map point predictions (BEV coordinates).

Used by UnifiedPerceptionDecoder for online map prediction supervision,
where each map element is represented as a sequence of 2D BEV points.
"""
import torch
import torch.nn as nn
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class PtsL1Loss(nn.Module):
    """L1 loss averaged over valid map points.

    Args:
        loss_weight (float): Weight of this loss term.
        reduction (str): 'mean' or 'sum'.
    """

    def __init__(self, loss_weight: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(
        self,
        pred_pts: torch.Tensor,
        gt_pts: torch.Tensor,
        gt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute L1 loss over map points.

        Args:
            pred_pts : (B, N_map, num_pts, 2)  predicted BEV waypoints
            gt_pts   : (B, N_map, num_pts, 2)  ground-truth BEV waypoints
            gt_mask  : (B, N_map) bool mask; True for valid map elements
        Returns:
            Scalar loss tensor.
        """
        loss = torch.abs(pred_pts - gt_pts)  # (B, N, P, 2)

        if gt_mask is not None:
            # Expand mask to cover pts and coords dimensions
            mask = gt_mask.unsqueeze(-1).unsqueeze(-1).float()  # (B, N, 1, 1)
            loss = (loss * mask).sum() / (mask.sum() * pred_pts.shape[-2] * pred_pts.shape[-1] + 1e-6)
        else:
            loss = loss.mean()

        return loss * self.loss_weight
