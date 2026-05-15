# Data Preparation — nuScenes Mini

This document describes how to download and prepare the **nuScenes mini** dataset
for use with UniDriveVLA.

## Dataset overview

| Property | nuScenes mini |
|----------|--------------|
| Version token | `v1.0-mini` |
| Scenes (train / val) | 7 / 3 |
| Samples (train / val) | ~282 / ~81 |
| Sensors | 6 cameras, 1 LiDAR, 5 radars |
| Download size | ~4 GB |

Because the mini split is tiny, results have high variance and should be treated
as a **sanity-check**, not a performance benchmark.

## Step 1 — Download nuScenes mini

1. Register (free) at [nuscenes.org](https://nuscenes.org/download).
2. Download **nuScenes v1.0 mini** (~4 GB).
3. Extract to `data/nuscenes/`:

```
data/
└── nuscenes/
    ├── v1.0-mini/
    │   ├── attribute.json
    │   ├── calibrated_sensor.json
    │   ├── category.json
    │   ├── ego_pose.json
    │   ├── instance.json
    │   ├── log.json
    │   ├── map.json
    │   ├── sample.json
    │   ├── sample_annotation.json
    │   ├── sample_data.json
    │   ├── scene.json
    │   ├── sensor.json
    │   └── visibility.json
    ├── maps/
    ├── samples/
    │   ├── CAM_FRONT/
    │   ├── CAM_FRONT_RIGHT/
    │   ├── CAM_FRONT_LEFT/
    │   ├── CAM_BACK/
    │   ├── CAM_BACK_LEFT/
    │   ├── CAM_BACK_RIGHT/
    │   └── LIDAR_TOP/
    └── sweeps/
```

## Step 2 — (Optional) Download CAN bus data

CAN bus data provides ego velocity / acceleration for the planning head.

1. Download **nuScenes CAN bus expansion** from [nuscenes.org/download](https://nuscenes.org/download).
2. Extract to `data/nuscenes/` (adds a `can_bus/` sub-directory).

If you skip this step, ego status will be zero-padded during training.

## Step 3 — Create info pkl files

```bash
cd nuScenes
bash scripts/create_data_mini.sh
```

This runs `tools/data_converter/nuscenes_converter.py` and produces:

```
data/infos/
├── nuscenes_infos_mini_train.pkl   # ~282 samples
└── nuscenes_infos_mini_val.pkl     #  ~81 samples
```

You can verify the pkl files:

```python
import pickle
with open('data/infos/nuscenes_infos_mini_train.pkl', 'rb') as f:
    infos = pickle.load(f)
print(f'Train samples: {len(infos)}')
print('Keys:', list(infos[0].keys()))
```

## Directory layout expected by configs

The training configs assume this layout relative to the `nuScenes/` directory:

```
nuScenes/
├── data/
│   ├── nuscenes/
│   │   ├── v1.0-mini/
│   │   ├── samples/
│   │   └── sweeps/
│   └── infos/
│       ├── nuscenes_infos_mini_train.pkl
│       └── nuscenes_infos_mini_val.pkl
├── projects/
├── tools/
└── scripts/
```

## Symlink alternative

If your nuScenes data lives elsewhere:

```bash
cd nuScenes
mkdir -p data
ln -s /path/to/your/nuscenes data/nuscenes
```

## Verifying the dataset

```bash
python - <<'EOF'
from nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=True)
print(f'Scenes : {len(nusc.scene)}')
print(f'Samples: {len(nusc.sample)}')
EOF
```

Expected output:
```
Scenes : 10
Samples: 404
```
