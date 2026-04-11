#!/usr/bin/env python3
"""
Roboshelf AI — Fázis 2: G1 Retail Navigáció PPO tanítás

Kaggle/Colab-kompatibilis tanítási script.
A G1 humanoid megtanul járni a retail boltban a raktárig.

Használat (Kaggle cellában):
  !python roboshelf_phase2_train.py --level teszt     # ~5 perc
  !python roboshelf_phase2_train.py --level kozepes   # ~30 perc
  !python roboshelf_phase2_train.py --level teljes    # ~2-4 óra
"""

import argparse
import os
import sys
import time
from pathlib import Path

# --- Import útvonal fix: envs mappa elérhetővé tétele ---
_THIS_DIR = Path(__file__).resolve().parent
_ENVS_DIR = _THIS_DIR.parent / "envs"
if str(_ENVS_DIR) not in sys.path:
    sys.path.insert(0, str(_ENVS_DIR))

import numpy as np
import gymnasium as gym

# --- Output mappa (Kaggle/Colab/lokális kompatibilis) ---
if os.path.exists("/kaggle/working"):
    OUTPUT_DIR = Path("/kaggle/working/roboshelf-phase2-results")
elif os.path.exists("/content"):
    OUTPUT_DIR = Path("/content/roboshelf-phase2-results")
else:
    # Repo-relatív útvonal: a script src/training/-ban van, tehát ../../roboshelf-results/phase2
    _REPO_ROOT = _THIS_DIR.parent.parent
    _RESULTS_DIR = _REPO_ROOT / "roboshelf-results" / "phase2"
    if _RESULTS_DIR.exists() or (_REPO_ROOT / "roboshelf-results").exists():
        OUTPUT_DIR = _RESULTS_DIR
    else:
        OUTPUT_DIR = Path.home() / "Documents" / "roboshelf-ai-dev" / "roboshelf-results" / "phase2"

MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"

# --- Szintek ---
LEVELS = {
    "teszt": {
        "total_timesteps": 100_000,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "Gyors teszt (~5 perc)",
    },
    "kozepes": {
        "total_timesteps": 2_000_000,
        "n_steps": 2048,
        "batch_size": 512,
        "n_epochs": 10,
        "n_envs": 8,
        "learning_rate": 1e-4,
        "clip_range": 0.15,
        "description": "Közepes (~30 perc GPU-n)",
    },
    "teljes": {
        "total_timesteps": 10_000_000,
        "n_steps": 2048,
        "batch_size": 1024,
        "n_epochs": 10,
        "n_envs": 16,
        "learning_rate": 1e-4,
        "clip_range": 0.15,
        "description": "Teljes (~2-4 óra GPU-n)",
    },
    "m2_2ora": {
        # M2 CPU-ra optimalizált ~2 órás tanítás
        # ~1000 FPS × 7200s = ~7.2M lépés reálisan, de 3M biztosan belefér
        "total_timesteps": 3_000_000,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "n_envs": 4,          # 4 párhuzamos env, CPU-n biztonságos
        "learning_rate": 1e-4,
        "clip_range": 0.15,
        "description": "M2 CPU ~2 óra (3M lépés, 4 env)",
    },
    "m2_6m": {
        # M2 CPU ~1 óra: 6M lépés, 4 env
        "total_timesteps": 6_000_000,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-5,   # kisebb LR a 3M-es modell folytatásához
        "clip_range": 0.1,
        "description": "M2 CPU ~1 óra (6M lépés, 4 env)",
    },
    "m2_3m_fresh": {
        # M2 CPU fresh start, ~2 óra: 3M lépés, optimalizált reward shaping
        # w_forward=4.0, w_healthy=3.0, w_fall=-10.0, w_gait=0.18 (env-ben beégetve)
        "total_timesteps": 3_000_000,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,   # magasabb LR a fresh starthoz
        "clip_range": 0.2,
        "description": "M2 CPU fresh start ~2 óra (3M lépés, új reward shaping)",
    },
    "m2_3m_nogait": {
        # M2 CPU fresh start, gait reward KIKAPCSOLVA (w_gait=0.0)
        # Tanulási sorrend: 1. járás megtanulása, 2. majd gait finom hangolás
        # w_forward=4.0, w_healthy=3.0, w_fall=-10.0, w_gait=0.0
        "total_timesteps": 3_000_000,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "M2 CPU fresh start ~2 óra (3M lépés, gait reward kikapcsolva)",
    },
    "m2_3m_v3": {
        # FIX: "stand and fall" probléma megoldása
        # w_forward=5.0 (domináns), w_healthy=0.5 (minimális), w_fall=-50.0 (erős)
        # Forrás: legged_gym + Gymnasium Humanoid-v4 tapasztalat
        "total_timesteps": 3_000_000,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "M2 CPU fresh start ~2 óra (stand-and-fall fix: w_healthy=0.5, w_fall=-50)",
    },
    "m2_3m_v4": {
        # FIX v4: w_healthy=0.0 (teljesen ki!), w_fall=-20 (mérsékelt), w_forward=5.0
        # v3 tanulság: w_fall=-50 túl agresszív → robot "befagyott" (ep hossz 35, reward -264)
        # v4: healthy nullán, fall mérsékelt → forward az egyetlen pozitív forrás
        "total_timesteps": 3_000_000,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "M2 CPU fresh start ~35 perc (v4: w_healthy=0.0, w_fall=-20, w_forward=5.0)",
    },
    "m2_3m_v5": {
        # FIX v5: helyes G1 kezdőpozíció! z=0.79 + kar joint szögek a keyframe alapján
        # Ez volt az igazi probléma: z=0.75 + karok rossz pozícióban → azonnal instabil
        # Reward: w_healthy=1.0 (mérsékelt), w_forward=5.0 (domináns), w_fall=-20
        "total_timesteps": 3_000_000,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "M2 CPU fresh start ~35 perc (v5: helyes G1 keyframe pozíció!)",
    },
    "m2_3m_v6": {
        # FIX v6: akció skálázás javítva!
        # Korábban: ctrl = ctrl_mean + action * ctrl_half (ctrl_mean rossz alap!)
        # Most: ctrl = default_ctrl + action * ctrl_half (keyframe az alap)
        # Nulla akció = egyensúlyi pozíció, nem random ctrl_mean
        "total_timesteps": 3_000_000,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "M2 CPU fresh start ~35 perc (v6: keyframe-alapú akció skálázás)",
    },
    "m2_3m_v7": {
        # FIX v7: sub-step 5→2, robot 100+ lépésen át stabil nulla akcióval
        # v6 tanulság: robot előre dőlt és 30 lépés alatt összecsuszik (5 sub-step túl gyors)
        # Most: 2 sub-step = lassabb fizika = több tanulási lehetőség
        "total_timesteps": 3_000_000,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "M2 CPU fresh start ~35 perc (v7: sub-step 5→2, stabil egyensúly)",
    },
}


def make_retail_env(n_envs=1, seed=42):
    """Retail nav env létrehozása VecNormalize-zel."""
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
    from stable_baselines3.common.utils import set_random_seed

    # Importáljuk a retail nav env-et
    from roboshelf_retail_nav_env import RoboshelfRetailNavEnv

    def make_single(rank):
        def _init():
            env = RoboshelfRetailNavEnv()
            env.reset(seed=seed + rank)
            return env
        return _init

    set_random_seed(seed)
    if n_envs > 1:
        env = SubprocVecEnv([make_single(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_single(0)])

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    return env


def make_eval_env():
    """Eval env létrehozása.

    Megjegyzés: a VecNormalize statisztikát a tanítási env-ből szinkronizáljuk
    az EvalCallback sync_vec_normalize=True opcióval, ezért itt training=False.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from roboshelf_retail_nav_env import RoboshelfRetailNavEnv

    env = DummyVecEnv([lambda: RoboshelfRetailNavEnv()])
    # norm_reward=False: eval-nál nem normalizáljuk a rewardot (valódi értékeket látunk)
    # training=False: a statisztika nem frissül eval közben
    env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
    return env


def train(args):
    """Fő tanítási loop."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    import torch

    cfg = LEVELS[args.level]
    timestamp = int(time.time())
    run_name = f"g1_retail_nav_{args.level}_{timestamp}"

    # Device meghatározás: CUDA > CPU
    # MPS (Apple Silicon) szándékosan kizárva: MlpPolicy float64-et használ,
    # amit az MPS framework nem támogat. CPU gyorsabb is MLP esetén.
    if torch.cuda.is_available():
        device = "cuda"
        device_label = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device = "cpu"
        device_label = "CPU (M2 optimalizált)"

    print(f"\n{'='*60}")
    print(f"  ROBOSHELF AI — Fázis 2: G1 Retail Navigáció")
    print(f"  Szint: {args.level} ({cfg['description']})")
    print(f"  Timesteps: {cfg['total_timesteps']:,}")
    print(f"  Envs: {cfg['n_envs']}, Batch: {cfg['batch_size']}")
    print(f"  Device: {device_label}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # Könyvtárak
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    best_dir = MODELS_DIR / "best"
    best_dir.mkdir(exist_ok=True)

    # Környezet
    print("  Környezetek létrehozása...")
    env = make_retail_env(n_envs=cfg["n_envs"])
    eval_env = make_eval_env()

    print(f"  ✅ {cfg['n_envs']}× RoboshelfRetailNav env kész")
    print(f"  Obs space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")

    # PPO modell
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=cfg["clip_range"],
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(LOGS_DIR),
        verbose=1,
        seed=42,
        device=device,
    )

    # VecNormalize szinkronizáló callback (train → eval stats másolás)
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecNormalize

    class SyncVecNormalizeCallback(BaseCallback):
        """Eval előtt szinkronizálja a VecNormalize statisztikát train env-ből."""
        def __init__(self, train_env, eval_env_ref):
            super().__init__()
            self.train_env = train_env
            self.eval_env_ref = eval_env_ref

        def _on_step(self):
            return True

        def on_eval_start(self):
            if isinstance(self.train_env, VecNormalize) and isinstance(self.eval_env_ref, VecNormalize):
                self.eval_env_ref.obs_rms = self.train_env.obs_rms
                self.eval_env_ref.ret_rms = self.train_env.ret_rms

    sync_cb = SyncVecNormalizeCallback(env, eval_env)

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(LOGS_DIR),
        eval_freq=max(cfg["total_timesteps"] // 20, 5000),
        n_eval_episodes=5,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(cfg["total_timesteps"] // 10, 10000),
        save_path=str(MODELS_DIR / "checkpoints"),
        name_prefix=run_name,
    )

    # Tanítás
    print(f"\n  🚀 Tanítás indítása...\n")
    start = time.time()

    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=[sync_cb, eval_callback, checkpoint_callback],
        tb_log_name=run_name,
        progress_bar=True,
    )

    elapsed = time.time() - start
    print(f"\n  ⏱️  Tanítás befejezve: {elapsed/60:.1f} perc ({elapsed/3600:.1f} óra)")

    # Mentés
    final_model = str(MODELS_DIR / f"{run_name}_final")
    model.save(f"{final_model}.zip")
    env.save(f"{final_model}_vecnormalize.pkl")
    env.save(str(best_dir / "best_model_vecnormalize.pkl"))
    print(f"  💾 Modell: {final_model}.zip")
    print(f"  💾 VecNormalize: {final_model}_vecnormalize.pkl")

    # Kiértékelés
    print(f"\n  📊 Kiértékelés...")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize as VN
    from roboshelf_retail_nav_env import RoboshelfRetailNavEnv

    ev = DummyVecEnv([lambda: RoboshelfRetailNavEnv()])
    ev = VN.load(f"{final_model}_vecnormalize.pkl", ev)
    ev.training = False
    ev.norm_reward = False

    rewards, lengths, dists = [], [], []
    for ep in range(10):
        obs = ev.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = ev.step(action)
            total_reward += reward[0]
            steps += 1
        rewards.append(total_reward)
        lengths.append(steps)
        if 'dist_to_target' in info[0]:
            dists.append(info[0]['dist_to_target'])
        print(f"    Ep {ep+1}: reward={total_reward:.1f}, lépés={steps}, táv={dists[-1] if dists else '?':.2f}m")

    print(f"\n    📈 Átlag reward: {np.mean(rewards):.1f} (±{np.std(rewards):.1f})")
    print(f"    📏 Átlag hossz: {np.mean(lengths):.0f}")
    if dists:
        print(f"    📍 Átlag távolság céltól: {np.mean(dists):.2f}m (start: 3.3m)")

    env.close()
    eval_env.close()
    ev.close()

    print(f"\n  ✅ Fájlok elmentve: {OUTPUT_DIR}")
    print(f"  Letöltés Kaggle-ből: Output fül → roboshelf-phase2-results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roboshelf AI — G1 Retail Nav PPO")
    parser.add_argument("--level", choices=list(LEVELS.keys()), default="teszt")
    args = parser.parse_args()
    train(args)
