"""Constants shared across UniDriveVLA dense-head modules."""

# nuScenes 3D detection categories (10 classes)
NUSCENES_DETECTION_CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

# nuScenes online-map element classes (3 classes)
NUSCENES_MAP_CLASSES = [
    "divider",
    "ped_crossing",
    "boundary",
]

# Shared spatial range (metres)
POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
VOXEL_SIZE = [0.2, 0.2, 8.0]

# nuScenes mini BEV grid (50×50, matches BEVFormer-tiny)
BEV_H_MINI = 50
BEV_W_MINI = 50

# Full nuScenes BEV grid
BEV_H_FULL = 200
BEV_W_FULL = 200

# Image normalisation (ImageNet statistics, used by BEVFormer & VAD)
IMG_NORM_MEAN = [123.675, 116.28, 103.53]
IMG_NORM_STD = [58.395, 57.12, 57.375]

# nuScenes mini image size  (H, W) — follows BEVFormer-tiny
INPUT_H_MINI = 450
INPUT_W_MINI = 800

# Number of cameras in nuScenes
NUM_CAMERAS = 6

# Planning horizon
PLANNING_STEPS = 6   # 3 s at 0.5 s intervals → 6 waypoints

# Motion-prediction horizon
MOTION_STEPS = 12    # 6 s at 0.5 s intervals

# Ego-status feature dimension
EGO_STATUS_DIM = 4   # (vx, vy, ax, ay)
