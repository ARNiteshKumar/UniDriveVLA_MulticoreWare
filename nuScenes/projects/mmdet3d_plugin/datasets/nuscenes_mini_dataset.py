"""
NuScenesMiniDataset
===================
Dataset class for nuScenes mini (v1.0-mini).

Key differences from the full nuScenes dataset:
- version = 'v1.0-mini'  (10 scenes: 7 train, 3 val)
- ~323 annotated samples
- queue_length = 2  (fewer temporal frames to avoid empty queues)
- Same 10 detection classes and 3 map element classes

The class extends mmdet3d's NuScenesDataset and overrides the temporal
queue loading to be robust to the mini split's small scene sizes.
"""

import os
import copy
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from mmdet.datasets import DATASETS

try:
    from mmdet3d.datasets import NuScenesDataset as _NuScenesBase
except ImportError:
    # Graceful fallback so the file can be imported without mmdet3d
    from torch.utils.data import Dataset as _NuScenesBase

from ..unidrivevla.dense_heads.constants import (
    NUSCENES_DETECTION_CLASSES,
    NUSCENES_MAP_CLASSES,
)


@DATASETS.register_module()
class NuScenesMiniDataset(_NuScenesBase):
    """nuScenes Mini dataset with temporal queuing for UniDriveVLA.

    Parameters
    ----------
    data_root : str
        Path to the nuScenes dataset root (contains 'v1.0-mini' folder).
    ann_file : str
        Path to the annotation pkl file.
    version : str
        Dataset version string.  Default: ``'v1.0-mini'``.
    queue_length : int
        Number of consecutive frames to stack in the temporal queue.
        Reduced to 2 for mini split to avoid empty queues.
    input_size : tuple[int, int]
        (H, W) of the resized camera images.
    use_can_bus : bool
        Whether to load CAN-bus ego-status signals.
    """

    CLASSES = tuple(NUSCENES_DETECTION_CLASSES)
    MAP_CLASSES = tuple(NUSCENES_MAP_CLASSES)

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        version: str = "v1.0-mini",
        queue_length: int = 2,
        input_size: Tuple[int, int] = (450, 800),
        use_can_bus: bool = True,
        overlap_test: bool = False,
        **kwargs,
    ):
        self.version = version
        self.queue_length = queue_length
        self.input_size = input_size
        self.use_can_bus = use_can_bus
        self.overlap_test = overlap_test

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            classes=list(self.CLASSES),
            **kwargs,
        )

        # Build scene → sample-index mapping for temporal queue construction
        self._build_scene_index()

    # ------------------------------------------------------------------
    # Indexing helpers
    # ------------------------------------------------------------------

    def _build_scene_index(self):
        """Group sample indices by scene token for temporal sampling."""
        self._scene_to_idxs: Dict[str, List[int]] = defaultdict(list)
        for idx, info in enumerate(self.data_infos):
            scene_token = info.get("scene_token", info.get("token", str(idx)))
            self._scene_to_idxs[scene_token].append(idx)

        # Sort by timestamp within each scene
        for token in self._scene_to_idxs:
            self._scene_to_idxs[token].sort(
                key=lambda i: self.data_infos[i].get("timestamp", i)
            )

        # Build reverse mapping: sample_idx → position within scene
        self._idx_to_scene: Dict[int, str] = {}
        self._idx_to_pos: Dict[int, int] = {}
        for token, idxs in self._scene_to_idxs.items():
            for pos, idx in enumerate(idxs):
                self._idx_to_scene[idx] = token
                self._idx_to_pos[idx] = pos

    def _get_queue_indices(self, idx: int) -> List[int]:
        """Return *queue_length* consecutive indices ending at *idx*.

        If the scene does not have enough history, the earliest available
        frame is repeated to fill the queue (left-pad strategy).
        """
        scene = self._idx_to_scene[idx]
        pos = self._idx_to_pos[idx]
        all_idxs = self._scene_to_idxs[scene]

        start = max(0, pos - self.queue_length + 1)
        slice_idxs = all_idxs[start : pos + 1]

        # Left-pad with the first available index if needed
        pad = self.queue_length - len(slice_idxs)
        if pad > 0:
            slice_idxs = [slice_idxs[0]] * pad + slice_idxs

        return slice_idxs[-self.queue_length :]

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def prepare_train_data(self, idx: int) -> Optional[Dict]:
        """Load a temporal queue of *queue_length* frames."""
        queue_idxs = self._get_queue_indices(idx)
        queue_data = []
        for qi in queue_idxs:
            data = self.prepare_single_frame(qi)
            if data is None:
                return None
            queue_data.append(data)

        # Stack queue into a single dict (last element is current frame)
        return self._stack_queue(queue_data)

    def prepare_test_data(self, idx: int) -> Optional[Dict]:
        """Same as train but without augmentation."""
        return self.prepare_train_data(idx)

    def prepare_single_frame(self, idx: int) -> Optional[Dict]:
        """Load a single frame.  Falls back to mmdet3d's default loader."""
        info = self.data_infos[idx]
        if self.test_mode:
            return self.pipeline(info)  # type: ignore[attr-defined]

        # Training augmentation pipeline (if configured)
        try:
            return self.pipeline(info)  # type: ignore[attr-defined]
        except Exception:
            return None

    def _stack_queue(self, queue: List[Dict]) -> Dict:
        """Merge a list of single-frame dicts into a temporal batch."""
        if not queue:
            return {}

        result = copy.deepcopy(queue[-1])  # use the last frame as the base

        # Stack image tensors along a new temporal dimension (dim=0 → T)
        if "img" in result and torch.is_tensor(result["img"]):
            imgs = [q["img"] for q in queue]
            result["img"] = torch.stack(imgs, dim=0)  # (T, N_cam, C, H, W)

        # Keep per-frame metadata as a list
        result["img_metas_queue"] = [q.get("img_metas", {}) for q in queue]

        return result

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, results, logger=None, **kwargs):
        """Delegate to nuScenes devkit evaluation."""
        try:
            return super().evaluate(results, logger=logger, **kwargs)
        except Exception as exc:
            if logger is not None:
                logger.warning(f"Evaluation failed: {exc}")
            return {}
