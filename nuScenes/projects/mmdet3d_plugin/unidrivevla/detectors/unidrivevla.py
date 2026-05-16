"""
UniDriveVLA top-level detector.

Wraps a QwenVL3APlanningHead (or any planning_head) and routes the
mmdet3d train/test protocol to it.  The class is intentionally thin;
all perception logic lives in the planning head and the
UnifiedPerceptionDecoder it owns.
"""

import torch
import torch.nn as nn
from mmdet.models import DETECTORS
from mmdet.models.builder import build_backbone, build_neck, build_head
from mmdet.models.detectors.base import BaseDetector


@DETECTORS.register_module()
class UniDriveVLA(BaseDetector):
    """Unified Vision-Language-Action detector for autonomous driving.

    Supports:
    - 3D object detection (10 nuScenes classes)
    - Online map prediction (3 element classes)
    - Ego-motion / trajectory planning
    - Optional AR co-training for driving VQA
    """

    def __init__(
        self,
        img_backbone=None,
        img_neck=None,
        planning_head=None,
        task_loss_weight=None,
        use_grid_mask=False,
        video_test_mode=False,
        **kwargs,
    ):
        super().__init__()
        if task_loss_weight is None:
            task_loss_weight = dict(planning=1.0)

        # Build backbone and neck (ResNet-50 + FPN, BEVFormer-tiny style)
        self.img_backbone = build_backbone(img_backbone) if img_backbone is not None else None
        self.img_neck = build_neck(img_neck) if img_neck is not None else None
        self.planning_head = build_head(planning_head) if planning_head is not None else None

        self.task_loss_weight = task_loss_weight
        self.use_grid_mask = use_grid_mask
        self.video_test_mode = video_test_mode
        self.instance_bank = None  # maintained inside heads (SparseDrive convention)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def with_planning_head(self):
        return self.planning_head is not None

    # ------------------------------------------------------------------
    # Required abstract methods from BaseDetector
    # ------------------------------------------------------------------

    def extract_feat(self, img):
        """Extract multi-camera image features via backbone + neck.

        Args:
            img: (B, N_cam, C, H, W) or (B, C, H, W)
        Returns:
            list of feature maps from the neck, or None if no backbone configured.
        """
        if self.img_backbone is None:
            return None

        # Handle multi-camera input: (B, N_cam, C, H, W)
        if img.dim() == 5:
            B, N, C, H, W = img.shape
            img = img.reshape(B * N, C, H, W)
            feats = self.img_backbone(img)
            if self.img_neck is not None:
                feats = self.img_neck(feats)
            # Reshape back to (B, N_cam, ...)
            feats = [f.reshape(B, N, *f.shape[1:]) for f in feats]
        else:
            feats = self.img_backbone(img)
            if self.img_neck is not None:
                feats = self.img_neck(feats)
        return feats

    def aug_test(self, imgs, **kwargs):
        return self.simple_test(imgs[0], **kwargs)

    def simple_test(self, img, **kwargs):
        if not self.with_planning_head:
            raise RuntimeError("planning_head is required for inference.")
        pred = self.planning_head.forward_test(img=img, **kwargs)
        return [{"img_bbox": pred}]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, return_loss=True, ar_batch=None, **kwargs):
        if return_loss:
            if ar_batch is None:
                ar_batch = getattr(self, "_current_ar_batch", None)
            return self.forward_train(ar_batch=ar_batch, **kwargs)
        return self.forward_test(**kwargs)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward_train(
        self,
        img=None,
        timestamp=None,
        projection_mat=None,
        image_wh=None,
        gt_depth=None,
        focal=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_map_labels=None,
        gt_map_pts=None,
        gt_agent_fut_trajs=None,
        gt_agent_fut_masks=None,
        gt_ego_fut_trajs=None,
        gt_ego_fut_masks=None,
        gt_ego_fut_cmd=None,
        ego_status=None,
        gt_occ_dense=None,
        hist_traj=None,
        ar_batch=None,
        **kwargs,
    ):
        if not self.with_planning_head:
            raise RuntimeError("planning_head is required for training.")

        img_last = self._select_last_from_queue(img)

        ret = self.planning_head.forward_train(
            img=img_last,
            timestamp=timestamp,
            projection_mat=projection_mat,
            image_wh=image_wh,
            gt_depth=gt_depth,
            focal=focal,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_map_labels=gt_map_labels,
            gt_map_pts=gt_map_pts,
            gt_agent_fut_trajs=gt_agent_fut_trajs,
            gt_agent_fut_masks=gt_agent_fut_masks,
            gt_ego_fut_trajs=gt_ego_fut_trajs,
            gt_ego_fut_masks=gt_ego_fut_masks,
            gt_ego_fut_cmd=gt_ego_fut_cmd,
            ego_status=ego_status,
            gt_occ_dense=gt_occ_dense,
            hist_traj=hist_traj,
            ar_batch=ar_batch,
            **kwargs,
        )

        raw_losses = ret["losses"] if isinstance(ret, dict) and "losses" in ret else ret
        losses = self._prefix_and_weight(raw_losses, prefix="planning")

        # Guard against NaN losses that can occur early in training
        device = img_last.device if torch.is_tensor(img_last) else torch.device("cuda")
        for k, v in list(losses.items()):
            if not torch.is_tensor(v):
                v = torch.tensor(v, device=device, dtype=torch.float32)
            losses[k] = torch.nan_to_num(v)

        return losses

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_test(
        self,
        img=None,
        timestamp=None,
        projection_mat=None,
        image_wh=None,
        gt_depth=None,
        focal=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_map_labels=None,
        gt_map_pts=None,
        gt_agent_fut_trajs=None,
        gt_agent_fut_masks=None,
        gt_ego_fut_trajs=None,
        gt_ego_fut_masks=None,
        gt_ego_fut_cmd=None,
        ego_status=None,
        gt_occ_dense=None,
        hist_traj=None,
        **kwargs,
    ):
        if not self.with_planning_head:
            raise RuntimeError("planning_head is required for testing.")

        img_last = self._select_last_from_queue(img)

        pred = self.planning_head.forward_test(
            img=img_last,
            timestamp=timestamp,
            projection_mat=projection_mat,
            image_wh=image_wh,
            gt_depth=gt_depth,
            focal=focal,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_map_labels=gt_map_labels,
            gt_map_pts=gt_map_pts,
            gt_agent_fut_trajs=gt_agent_fut_trajs,
            gt_agent_fut_masks=gt_agent_fut_masks,
            gt_ego_fut_trajs=gt_ego_fut_trajs,
            gt_ego_fut_masks=gt_ego_fut_masks,
            gt_ego_fut_cmd=gt_ego_fut_cmd,
            ego_status=ego_status,
            hist_traj=hist_traj,
            **kwargs,
        )

        if isinstance(pred, dict) and ("det" in pred or "map" in pred):
            return self._unpack_multitask_pred(pred)

        # Fallback: planning-only output
        return [{"img_bbox": {"trajs_3d": pred}}]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_last_from_queue(self, img):
        """Handle (B, T, N_cam, C, H, W) temporal queue; return last frame."""
        if torch.is_tensor(img) and img.dim() == 6:
            return img[:, -1]
        return img

    def _prefix_and_weight(self, loss_dict, prefix):
        factor = float(self.task_loss_weight.get(prefix, 1.0))
        return {f"{prefix}.{k}": v * factor for k, v in loss_dict.items()}

    def _unpack_multitask_pred(self, pred):
        traj = pred.get("planning", pred.get("traj"))
        det_list = pred.get("det")
        map_list = pred.get("map")

        if torch.is_tensor(traj) and traj.dim() == 3:
            batch_size = traj.shape[0]
        elif isinstance(det_list, list):
            batch_size = len(det_list)
        elif isinstance(map_list, list):
            batch_size = len(map_list)
        else:
            batch_size = 1

        outputs = []
        for i in range(int(batch_size)):
            img_bbox = {}
            if isinstance(det_list, list) and i < len(det_list) and isinstance(det_list[i], dict):
                img_bbox.update(det_list[i])
            if isinstance(map_list, list) and i < len(map_list) and isinstance(map_list[i], dict):
                img_bbox.update(map_list[i])
            if torch.is_tensor(traj) and traj.dim() == 3:
                img_bbox["final_planning"] = traj.detach().cpu()[i]
            elif traj is not None:
                img_bbox["final_planning"] = traj
            outputs.append({"img_bbox": img_bbox})

        return outputs
