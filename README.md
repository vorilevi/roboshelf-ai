# Roboshelf AI

**Humanoid robot training for retail shelf stocking**

AI-driven humanoid robot training pipeline using MuJoCo simulation, PPO reinforcement learning, and NVIDIA GR00T N1 foundation model — targeting autonomous retail shelf restocking.

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ In Progress | MuJoCo Humanoid-v5 locomotion training |
| Phase 2 | ⬜ Planned | Retail environment modeling (MJCF) |
| Phase 3 | ⬜ Planned | Task-specific RL (grasping, placement) |
| Phase 4 | ⬜ Planned | Hierarchical policy integration |
| Phase 5 | ⬜ Planned | Investor demo |

## Tech Stack

- **Simulation**: MuJoCo 3.6+ / MJX / MuJoCo Playground
- **RL Framework**: Stable-Baselines3 (PPO)
- **ML**: PyTorch, JAX
- **Foundation Model**: NVIDIA GR00T N1.6 (planned)
- **Monitoring**: TensorBoard, Weights & Biases
- **Hardware**: MacBook Air M2 (dev) → Cloud GPU (training)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run system diagnostics
bash src/roboshelf_system_check.sh

# Train (quick test, ~5 min)
python src/training/roboshelf_phase1_train.py

# Train (medium, ~1-2 hours)
python src/training/roboshelf_phase1_train.py --level kozepes

# Watch trained agent
python src/training/roboshelf_phase1_train.py --watch

# TensorBoard monitoring
tensorboard --logdir=~/roboshelf-results/logs
```

## Project Structure

```
roboshelf-ai/
├── src/
│   ├── training/          # RL training scripts
│   ├── envs/              # Custom MuJoCo environments (Phase 2+)
│   └── evaluation/        # Evaluation and benchmarking
├── models/                # Saved model checkpoints
├── notebooks/             # Jupyter notebooks (experiments)
├── docs/                  # Documentation and plans
└── requirements.txt
```

## License

Proprietary — All rights reserved.

## Contact

Levente Vöröss — [vorilevi@gmail.com](mailto:vorilevi@gmail.com)
