#!/usr/bin/env python3
"""
Humanoid-v4 baseline tanítás SB3 PPO-val.

Cél: meggyőződni hogy az SB3 PPO + MuJoCo Humanoid tud tanulni járni
mielőtt a G1 retail env-et javítjuk tovább.

Referencia eredmény (community):
- 1M lépésnél: reward ~1000-3000
- 3M lépésnél: reward ~3000-6000
- Ha ez nem jön össze → SB3/MuJoCo konfig probléma

Használat:
  python src/training/humanoid_v4_baseline.py
  python src/training/humanoid_v4_baseline.py --steps 5000000
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

# Output mappa
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
OUTPUT_DIR = _REPO_ROOT / "roboshelf-results" / "humanoid_v4_baseline"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR   = OUTPUT_DIR / "logs"


def make_env(rank, seed=42):
    def _init():
        env = gym.make("Humanoid-v4")
        env.reset(seed=seed + rank)
        return env
    return _init


def make_eval_env():
    env = gym.make("Humanoid-v4")
    from stable_baselines3.common.vec_env import DummyVecEnv
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, training=False)
    return venv


def train(args):
    timestamp = int(time.time())
    run_name  = f"humanoid_v4_{timestamp}"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    best_dir = MODELS_DIR / "best"
    best_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Humanoid-v4 Baseline — SB3 PPO")
    print(f"  Timesteps: {args.steps:,}")
    print(f"  Envs: 4, Device: CPU")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # Env
    set_random_seed(42)
    env = SubprocVecEnv([make_env(i) for i in range(4)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True,
                       clip_obs=10.0, clip_reward=10.0)

    eval_env = make_eval_env()

    print(f"  Obs space:    {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")

    # PPO — Humanoid-v4 bevált konfig (SB3 zoo alapján)
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(LOGS_DIR),
        verbose=1,
        seed=42,
        device="cpu",
    )

    eval_freq = max(args.steps // 20, 50_000)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(LOGS_DIR),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.steps // 5, 100_000),
        save_path=str(MODELS_DIR / "checkpoints"),
        name_prefix=run_name,
    )

    # Tanítás
    print(f"\n  🚀 Tanítás indítása...\n")
    start = time.time()

    model.learn(
        total_timesteps=args.steps,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name=run_name,
        progress_bar=True,
    )

    elapsed = time.time() - start
    print(f"\n  ⏱️  Befejezve: {elapsed/60:.1f} perc")

    # Mentés
    final = str(MODELS_DIR / f"{run_name}_final")
    model.save(f"{final}.zip")
    env.save(f"{final}_vecnormalize.pkl")
    print(f"  💾 {final}.zip")

    # Kiértékelés — 10 epizód
    print(f"\n  📊 Kiértékelés (10 ep)...")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize as VN
    ev = DummyVecEnv([lambda: gym.make("Humanoid-v4")])
    ev = VN.load(f"{final}_vecnormalize.pkl", ev)
    ev.training = False
    ev.norm_reward = False

    rewards, lengths = [], []
    for ep in range(10):
        obs = ev.reset()
        done, total_r, steps = False, 0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, info = ev.step(action)
            total_r += r[0]
            steps += 1
        rewards.append(total_r)
        lengths.append(steps)
        print(f"    Ep {ep+1:2d}: reward={total_r:8.1f}, lépés={steps}")

    print(f"\n    📈 Átlag reward: {np.mean(rewards):.1f} (±{np.std(rewards):.1f})")
    print(f"    📏 Átlag hossz:  {np.mean(lengths):.0f}")

    # Értelmezés
    mean_r = np.mean(rewards)
    print(f"\n  {'='*50}")
    if mean_r > 3000:
        print(f"  ✅ KIVÁLÓ: reward={mean_r:.0f} — SB3 PPO + MuJoCo rendben")
        print(f"     → G1 env-ben van a probléma, nem az SB3 konfigban")
    elif mean_r > 1000:
        print(f"  ✅ JÓ: reward={mean_r:.0f} — tanulás folyamatban")
        print(f"     → Több lépéssel (5-10M) éri el a csúcsot")
    elif mean_r > 0:
        print(f"  ⚠️  RÉSZLEGES: reward={mean_r:.0f} — tanul, de lassú")
        print(f"     → Ellenőrizd a MuJoCo verziót és az SB3 zoo konfigját")
    else:
        print(f"  ❌ PROBLÉMA: reward={mean_r:.0f} — nem tanul")
        print(f"     → Ellenőrizd: pip show mujoco stable-baselines3")
    print(f"  {'='*50}")

    env.close()
    eval_env.close()
    ev.close()

    print(f"\n  TensorBoard: tensorboard --logdir {LOGS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3_000_000)
    args = parser.parse_args()
    train(args)
