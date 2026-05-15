"""
NuScenes3DDataset — adapted for nuScenes mini (v1.0-mini).

Key differences from the full-dataset version:
  - version = 'v1.0-mini'
  - queue_length = 2  (2 temporal frames instead of 4)
  - Loads from nuscenes_infos_mini_train.pkl / nuscenes_infos_mini_val.pkl
  - Smaller BEV grid: 50×50

The dataset returns dicts that are consumed by the mmdet3d dataloader and
forwarded to UniDriveVLA.forward_train / forward_test.
"""

from __future__ import annotations

import copy
import math
import os
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset

# Re-export for convenience
from ..unidrivevla.dense_heads.constants import (
    NUSCENES_DETECTION_CLASSES,
    NUSCENES_MAP_CLASSES,
    BEV_H_MINI,
    BEV_W_MINI,
    PLANNING_STEPS,
    EGO_STATUS_DIM,
)


@DATASETS.register_module()
class NuScenes3DDataset(NuScenesDataset):
    """Multi-task nuScenes dataset for UniDriveVLA.

    Extends the standard mmdet3d NuScenesDataset with:
    - Temporal queue (queue_length frames of history)
    - Online map ground truth
    - Ego trajectory (history + future)
    - Agent future trajectories
    - Command tokens

    Parameters
    ----------
    queue_length : int
        Number of frames in the temporal queue (2 for mini, 4 for full).
    bev_size : tuple[int, int]
        (H, W) of the BEV grid.
    overlap_test : bool
        Whether to allow sample overlap in test mode.
    version : str
        nuScenes version string ('v1.0-mini' or 'v1.0').
    map_classes : list[str]
        Map element class names.
    """

    CLASSES = tuple(NUSCENES_DETECTION_CLASSES)
    MAP_CLASSES = tuple(NUSCENES_MAP_CLASSES)

    def __init__(
        self,
        queue_length: int = 2,
        bev_size: Tuple[int, int] = (BEV_H_MINI, BEV_W_MINI),
        overlap_test: bool = False,
        version: str = 'v1.0-mini',
        map_classes: Optional[List[str]] = None,
        point_cloud_range: Optional[List[float]] = None,
        **kwargs,
    ):
        # Override version if passed (mini vs full)
        kwargs.setdefault('version', version)
        super().__init__(**kwargs)

        self.queue_length = queue_length
        self.bev_size = bev_size
        self.overlap_test = overlap_test
        self.map_classes = map_classes or list(NUSCENES_MAP_CLASSES)
        self.point_cloud_range = point_cloud_range or [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

        # Build scene-level index for temporal queue sampling
        self._build_scene_index()

    # ------------------------------------------------------------------
    # Scene index helpers
    # ------------------------------------------------------------------

    def _build_scene_index(self):
        """Map sample tokens to their scene and frame position.

        Builds:
            self.scene_to_samples: dict[scene_token -> list[data_info_idx]]
            self.sample_to_prev:   dict[data_info_idx -> data_info_idx | None]
        """
        self.scene_to_samples: Dict[str, List[int]] = {}
        self.sample_to_prev: Dict[int, Optional[int]] = {}

        for idx, info in enumerate(self.data_infos):
            scene_token = info.get('scene_token', f'scene_{idx}')
            if scene_token not in self.scene_to_samples:
                self.scene_to_samples[scene_token] = []
            pos = len(self.scene_to_samples[scene_token])
            self.scene_to_samples[scene_token].append(idx)
            if pos == 0:
                self.sample_to_prev[idx] = None
            else:
                self.sample_to_prev[idx] = self.scene_to_samples[scene_token][pos - 1]

    def _get_queue(self, idx: int) -> List[int]:
        """Return list of indices for the temporal queue ending at *idx*.

        Returns at most queue_length indices (older frames first).
        """
        queue: List[int] = [idx]
        cur = idx
        for _ in range(self.queue_length - 1):
            prev = self.sample_to_prev.get(cur)
            if prev is None:
                break
            queue.insert(0, prev)
            cur = prev
        # Pad with earliest available frame if queue is shorter
        while len(queue) < self.queue_length:
            queue.insert(0, queue[0])
        return queue

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def prepare_train_data(self, index: int) -> Optional[Dict]:
        """Load and pipeline-process a training sample with its queue."""
        queue_indices = self._get_queue(index)
        queue_data = []
        for qi in queue_indices:
            input_dict = self.get_data_info(qi)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            try:
                example = self.pipeline(input_dict)
            except Exception:
                return None
            if example is None:
                return None
            queue_data.append(example)
        return self._collect_queue(queue_data)

    def prepare_test_data(self, index: int) -> Dict:
        """Load and pipeline-process a test sample (no queue)."""
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    # ------------------------------------------------------------------
    # Queue collation
    # ------------------------------------------------------------------

    def _collect_queue(self, queue: List[Dict]) -> Dict:
        """Merge list of per-frame dicts into a single batched dict.

        Image tensors are stacked along a new temporal dimension (dim=0).
        Annotations come from the *last* (current) frame.
        """
        if len(queue) == 1:
            return queue[-1]

        last = queue[-1]
        # Stack images along temporal axis (T, N_cam, C, H, W)
        imgs = [q['img'].data for q in queue]  # each: (N_cam, C, H, W)
        try:
            import torch
            stacked_img = torch.stack(imgs, dim=0)
            last['img'] = type(last['img'])(stacked_img)
        except Exception:
            pass  # fallback: return last frame only

        return last

    # ------------------------------------------------------------------
    # get_data_info override to add ego/map info
    # ------------------------------------------------------------------

    def get_data_info(self, index: int) -> Optional[Dict]:
        info = self.data_infos[index]
        input_dict = super().get_data_info(index)
        if input_dict is None:
            return None

        # Add ego trajectory information (placeholder; real impl reads canbus)
        input_dict['ego_status'] = self._get_ego_status(info)
        input_dict['gt_ego_fut_trajs'] = self._get_ego_fut_trajs(info)
        input_dict['gt_ego_fut_masks'] = np.ones(PLANNING_STEPS, dtype=np.float32)
        input_dict['gt_ego_fut_cmd'] = info.get('command', 0)

        # Map ground truth (placeholder; real impl uses map_api)
        input_dict['gt_map_labels'] = []
        input_dict['gt_map_pts'] = []

        return input_dict

    def _get_ego_status(self, info: Dict) -> np.ndarray:
        """Return (vx, vy, ax, ay) ego status from info dict."""
        can = info.get('can_bus', None)
        if can is not None and len(can) >= 4:
            return np.array(can[:EGO_STATUS_DIM], dtype=np.float32)
        return np.zeros(EGO_STATUS_DIM, dtype=np.float32)

    def _get_ego_fut_trajs(self, info: Dict) -> np.ndarray:
        """Return (PLANNING_STEPS, 2) future ego waypoints."""
        trajs = info.get('gt_ego_fut_trajs', None)
        if trajs is not None:
            t = np.array(trajs, dtype=np.float32)
            if t.shape[0] >= PLANNING_STEPS:
                return t[:PLANNING_STEPS, :2]
        return np.zeros((PLANNING_STEPS, 2), dtype=np.float32)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, results, metric='bbox', **kwargs):
        """Forward to nuScenes official evaluator."""
        return super().evaluate(results, metric=metric, **kwargs)
