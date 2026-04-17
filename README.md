# Roboshelf AI

**Humanoid robot training for retail shelf stocking**

AI-driven humanoid robot training pipeline using MuJoCo simulation and PPO reinforcement learning — targeting autonomous retail shelf restocking. The Unitree G1 humanoid learns to navigate a retail store environment and eventually restock shelves.

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Done | Humanoid-v5 locomotion baseline (reward=855 @ 3M steps) |
| Phase 2 | 🔄 Training | G1 retail navigation: start → warehouse (3.3m), v22 running |
| Phase 3 | 📝 Drafted | Pick & place manipulation (env written, training pending) |
| Phase 4 | ⬜ Planned | Hierarchical policy integration |
| Phase 5 | ⬜ Planned | Investor demo |

## Phase 2 Progress

The G1 robot learns to walk from the store entrance (y=0.5) to the warehouse (y=3.8) — a 3.3m navigation task.

| Version | Reward | Ep Length | Notes |
|---------|--------|-----------|-------|
| v17 | +199 @ 8M | — | Curriculum worked, then thrashing kicked in |
| v18 | stable | 166 steps | Robot goes *backwards* (w_dof_vel blocked movement) |
| v19 | -1281 | 31 steps | Regression: hip lean destabilized stance |
| **v20** | **+126** | **138 steps** | **Breakthrough: robot moves forward!** |
| v21 | -317 | 86 steps | Fine-tune failed: standing still is more rewarding than moving |
| **v22** | 🔄 running | — | Fresh start: bent-knee stance + foot slip/distance penalties + reward rebalance |

**Key insight (v22):** Three physical fixes from Unitree's official `unitree_rl_gym`:
1. **Bent-knee default pose** — `hip_pitch=-0.1`, `knee=+0.3`, `ankle_pitch=-0.2` rad. Straight legs make it physically impossible to lift a leg without toppling.
2. **Foot slip penalty** — penalizes the stance foot sliding backward, ensuring stable ground contact.
3. **Foot distance penalty** — prevents the robot from crossing its own feet and tripping itself.

## Tech Stack

- **Simulation**: MuJoCo 3.6+
- **RL Framework**: Stable-Baselines3 (PPO)
- **Robot**: Unitree G1 (29 DoF, 35 kg)
- **ML**: PyTorch
- **Monitoring**: TensorBoard
- **Hardware**: MacBook Air M2 (dev + training)

## Quick Start

```bash
# Install dependencies
pip install mujoco gymnasium stable-baselines3 torch

# Run Phase 2 training (fresh start, ~2 hours on M2)
cd roboshelf-ai
python src/training/roboshelf_phase2_train.py --level m2_20m_v22

# Watch trained agent (macOS: mjpython required!)
mjpython replay_policy.py --slowdown 2.0

# TensorBoard monitoring
tensorboard --logdir roboshelf-results/phase2/logs
```

## Project Structure

```
roboshelf-ai/
├── src/
│   ├── training/
│   │   ├── roboshelf_phase2_train.py     # Phase 2 training (active: m2_20m_v22)
│   │   └── roboshelf_phase2_finetune.py  # Fine-tune from existing model
│   └── envs/
│       ├── roboshelf_retail_nav_env.py   # G1 retail navigation env (v22)
│       ├── roboshelf_manipulation_env.py # Phase 3 env (drafted)
│       └── assets/
│           └── roboshelf_retail_store.xml # Store MJCF scene
├── replay_policy.py                      # Policy visualization
├── roboshelf-results/phase2/
│   ├── models/                           # Saved checkpoints
│   └── logs/                             # TensorBoard logs
└── CONTEXT.md                            # AI session context (internal)
```

## License

Proprietary — All rights reserved.

## Contact

Levente Vöröss — [vorilevi@gmail.com](mailto:vorilevi@gmail.com)
