# Bench2Drive Setup (Requires CARLA 0.9.15)

> **Important**: Bench2Drive closed-loop evaluation requires a **local CARLA installation**.
> It cannot run in standard CI or on a machine without CARLA.
> The GitHub Actions workflow (`bench2drive_eval.yml`) only performs code
> correctness checks — not actual simulation.

## Overview

Bench2Drive is a closed-loop autonomous driving benchmark that runs UniDriveVLA
as a planning agent inside the CARLA simulator. The agent receives camera images
from CARLA, runs the model forward pass, and outputs control commands.

## Prerequisites

| Requirement | Version |
|-------------|---------|
| CARLA       | 0.9.15  |
| Python      | 3.10    |
| GPU memory  | ≥ 24 GB |
| OS          | Ubuntu 20.04 / 22.04 |
| Disk space  | ≥ 50 GB |

## Step 1 — Install CARLA 0.9.15

```bash
# Download CARLA 0.9.15
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xzf CARLA_0.9.15.tar.gz -C /opt/carla

# Install CARLA Python API
pip install /opt/carla/PythonAPI/carla/dist/carla-0.9.15-cp310-cp310-linux_x86_64.whl
```

## Step 2 — Install Bench2Drive

```bash
git clone https://github.com/Thinklab-SJTU/Bench2Drive.git
cd Bench2Drive
pip install -r requirements.txt
```

## Step 3 — Configure UniDriveVLA Agent

Copy the UniDriveVLA CARLA agent wrapper to Bench2Drive:

```bash
cp tools/bench2drive/unidrivevla_agent.py Bench2Drive/team_code/
```

Set the model checkpoint in the agent config:

```bash
export UNIDRIVEVLA_CKPT="work_dirs/mini_stage2/latest.pth"
export UNIDRIVEVLA_CFG="projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py"
```

## Step 4 — Start CARLA Server

In a separate terminal:

```bash
/opt/carla/CarlaUE4.sh -RenderOffScreen -world-port=2000 -carla-rpc-port=2000
```

Wait for the CARLA server to be ready (typically 30–60 s).

## Step 5 — Run Bench2Drive Evaluation

```bash
cd Bench2Drive

python leaderboard/scripts/run_evaluation.py \
    --agent team_code/unidrivevla_agent.py \
    --agent-config "${UNIDRIVEVLA_CFG}" \
    --checkpoint "${UNIDRIVEVLA_CKPT}" \
    --host localhost \
    --port 2000 \
    --trafficManagerPort 8000 \
    --routes leaderboard/data/bench2drive42/bench2drive_routes_42.xml \
    --repetitions 1 \
    --output-dir results/bench2drive
```

## Metrics

Bench2Drive reports:
- **DS** — Driving Score (composite)
- **RC** — Route Completion (%)
- **IS** — Infraction Score
- **Collision** — Collisions with pedestrians/vehicles/layout

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CARLA won't start | Check GPU driver; try `DISPLAY=` prefix |
| Segmentation fault | Downgrade CARLA to 0.9.14 or update GPU driver |
| Agent timeout | Reduce image resolution in agent config |
| OOM on GPU | Use `--fp16` or reduce batch size |

## Notes on nuScenes Mini

The nuScenes mini model checkpoint was trained on a very small dataset.
Bench2Drive performance will be substantially lower than a model trained on
the full nuScenes dataset. Mini weights are intended for **debugging the
pipeline**, not for benchmark submission.
