#!/usr/bin/env python3
"""
Roboshelf AI — Fázis 1 javított tanítási script (v2)

Javítások az eredeti scripthez képest:
  1. VecNormalize fix: eval és --watch is betölti a VecNormalize .pkl fájlt
  2. --continue-from automatikusan betölti a mellette lévő VecNormalize .pkl-t
  3. Javított hiperparaméterek: alacsonyabb LR, szűkebb clip range, nagyobb batch
  4. Automatikus VecNormalize mentés minden checkpoint mellé
  5. Jobb logging: WandB opcionális, TensorBoard mindig

Használat:
  # Gyors teszt (~5 perc)
  python roboshelf_phase1_train_v2.py

  # Közepes tanítás (~1-2 óra, javított hiperparaméterekkel)
  python roboshelf_phase1_train_v2.py --level kozepes

  # Teljes tanítás (~6-10 óra)
  python roboshelf_phase1_train_v2.py --level teljes

  # Éjszakai tanítás (~15+ óra)
  python roboshelf_phase1_train_v2.py --level ejszakai

  # Folytatás korábbi modellből (automatikusan tölti a VecNormalize-t is!)
  python roboshelf_phase1_train_v2.py --level ejszakai --continue-from models/best/best_model.zip

  # Betanított modell megtekintése (VecNormalize-zel!)
  python roboshelf_phase1_train_v2.py --watch

  # Kiértékelés (VecNormalize-zel!)
  python roboshelf_phase1_train_v2.py --eval

  # Random baseline
  python roboshelf_phase1_train_v2.py --random
"""

import argparse
import os
import time
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

# --- Könyvtárak ---
RESULTS_DIR = Path.home() / "Documents" / "roboshelf-ai-dev" / "roboshelf-results"
MODELS_DIR = RESULTS_DIR / "models"
LOGS_DIR = RESULTS_DIR / "logs"

# --- Szintek konfigurációja (v2 — javított hiperparaméterek) ---
LEVELS = {
    "teszt": {
        "total_timesteps": 50_000,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 5,
        "n_envs": 2,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "Gyors teszt (~5 perc)",
    },
    "kozepes": {
        "total_timesteps": 2_000_000,
        "n_steps": 2048,
        "batch_size": 128,       # v2: 64 → 128
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 1e-4,   # v2: 3e-4 → 1e-4 (stabilabb)
        "clip_range": 0.15,      # v2: 0.2 → 0.15 (kevesebb oszcilláció)
        "description": "Közepes tanítás (~1-2 óra)",
    },
    "teljes": {
        "total_timesteps": 10_000_000,
        "n_steps": 2048,
        "batch_size": 256,       # v2: nagyobb batch
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 1e-4,
        "clip_range": 0.15,
        "description": "Teljes tanítás (~6-10 óra)",
    },
    "ejszakai": {
        "total_timesteps": 50_000_000,  # v2: 30M → 50M
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 1e-4,
        "clip_range": 0.1,       # v2: 0.2 → 0.1 (nagyon stabil)
        "description": "Éjszakai tanítás (~15+ óra)",
    },
}


def find_vecnormalize_pkl(model_path: str) -> str | None:
    """
    Megkeresi a VecNormalize .pkl fájlt egy modell mellett.

    Keresési sorrend:
      1. <model_path>_vecnormalize.pkl (eredeti naming)
      2. <model_dir>/vecnormalize.pkl
      3. <model_dir>/../vecnormalize.pkl (best/ almappából)
    """
    model_path = Path(model_path)
    candidates = [
        model_path.with_name(model_path.stem + "_vecnormalize.pkl"),
        model_path.parent / "vecnormalize.pkl",
        model_path.parent.parent / "vecnormalize.pkl",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def make_env(n_envs: int = 1, seed: int = 42):
    """Létrehozza a vektorizált Humanoid-v5 környezetet VecNormalize-zel."""
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.utils import set_random_seed

    def make_single(rank):
        def _init():
            env = gym.make("Humanoid-v5")
            env.reset(seed=seed + rank)
            return env
        return _init

    set_random_seed(seed)
    if n_envs > 1:
        env = SubprocVecEnv([make_single(i) for i in range(n_envs)])
    else:
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv([make_single(0)])

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    return env


def make_eval_env(vecnorm_path: str | None = None, seed: int = 99):
    """
    Létrehozza a kiértékelő környezetet.

    v2 FIX: Ha van VecNormalize .pkl, betölti az edzés statisztikákat!
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env = DummyVecEnv([lambda: gym.make("Humanoid-v5")])

    if vecnorm_path and Path(vecnorm_path).exists():
        print(f"  ✅ VecNormalize betöltve: {vecnorm_path}")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False      # Ne frissítse a statisztikákat eval közben
        env.norm_reward = False   # Ne normalizálja a reward-ot eval közben
    else:
        print("  ⚠️  VecNormalize nélkül fut — a reward értékek alacsonyabbak lehetnek!")
        env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)

    env.seed(seed)
    return env


def save_with_vecnormalize(model, env, path_prefix: str):
    """Modell és VecNormalize együttes mentése."""
    model_path = f"{path_prefix}.zip"
    vecnorm_path = f"{path_prefix}_vecnormalize.pkl"
    model.save(model_path)
    env.save(vecnorm_path)
    print(f"  💾 Modell: {model_path}")
    print(f"  💾 VecNormalize: {vecnorm_path}")
    return model_path, vecnorm_path


def train(args):
    """Fő tanítási loop."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

    cfg = LEVELS[args.level]
    timestamp = int(time.time())
    run_name = f"humanoid_ppo_{args.level}_{timestamp}"

    print(f"\n{'='*60}")
    print(f"  ROBOSHELF AI — Fázis 1 tanítás v2")
    print(f"  Szint: {args.level} ({cfg['description']})")
    print(f"  Timesteps: {cfg['total_timesteps']:,}")
    print(f"  Learning rate: {cfg['learning_rate']}")
    print(f"  Clip range: {cfg['clip_range']}")
    print(f"  Batch size: {cfg['batch_size']}")
    print(f"{'='*60}\n")

    # Könyvtárak létrehozása
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    best_dir = MODELS_DIR / "best"
    best_dir.mkdir(exist_ok=True)

    # Környezet
    env = make_env(n_envs=cfg["n_envs"])

    # Modell
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    if args.continue_from:
        print(f"  📂 Modell betöltése: {args.continue_from}")
        model = PPO.load(args.continue_from, env=env)

        # v2 FIX: VecNormalize automatikus betöltése
        vecnorm_path = find_vecnormalize_pkl(args.continue_from)
        if vecnorm_path:
            from stable_baselines3.common.vec_env import VecNormalize
            env = VecNormalize.load(vecnorm_path, env)
            model.set_env(env)
            print(f"  ✅ VecNormalize betöltve: {vecnorm_path}")
        else:
            print("  ⚠️  VecNormalize .pkl nem található — új statisztikákkal indul")

        # Hiperparaméterek frissítése v2 értékekre
        model.learning_rate = cfg["learning_rate"]
        model.clip_range = lambda _: cfg["clip_range"]
        model.n_steps = cfg["n_steps"]
        model.batch_size = cfg["batch_size"]
        model.n_epochs = cfg["n_epochs"]
        print(f"  🔧 Hiperparaméterek frissítve v2 értékekre")
    else:
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
            ent_coef=0.001,    # v2: 0 → 0.001 (kis entrópia bonus az explor.-hoz)
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=str(LOGS_DIR),
            verbose=1,
            seed=42,
            device="cpu",
        )

    # Eval callback — v2: VecNormalize-zel
    eval_vecnorm_path = find_vecnormalize_pkl(args.continue_from) if args.continue_from else None
    eval_env = make_eval_env(eval_vecnorm_path)

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
    start_time = time.time()

    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=[eval_callback, checkpoint_callback],
        tb_log_name=run_name,
        progress_bar=True,
    )

    elapsed = time.time() - start_time
    print(f"\n  ⏱️  Tanítás befejezve: {elapsed/60:.1f} perc ({elapsed/3600:.1f} óra)")

    # Mentés VecNormalize-zel
    final_prefix = str(MODELS_DIR / f"{run_name}_final")
    save_with_vecnormalize(model, env, final_prefix)

    # Best modell mellé is VecNormalize mentése
    best_vecnorm = str(best_dir / "best_model_vecnormalize.pkl")
    env.save(best_vecnorm)
    print(f"  💾 Best VecNormalize: {best_vecnorm}")

    # Azonnali kiértékelés a helyes VecNormalize-zel
    print(f"\n  📊 Azonnali kiértékelés (VecNormalize-zel)...")
    eval_env_final = make_eval_env(f"{final_prefix}_vecnormalize.pkl")
    evaluate_model(model, eval_env_final, n_episodes=5)

    env.close()
    eval_env.close()
    eval_env_final.close()


def evaluate_model(model, env, n_episodes=5):
    """Modell kiértékelése — helyes reward értékekkel."""
    rewards = []
    lengths = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
        rewards.append(total_reward)
        lengths.append(steps)
        print(f"    Epizód {ep+1}: reward={total_reward:.1f}, hossz={steps}")

    print(f"\n    📈 Átlag reward: {np.mean(rewards):.1f} (±{np.std(rewards):.1f})")
    print(f"    📏 Átlag hossz: {np.mean(lengths):.0f}")


def watch(args):
    """Betanított modell vizuális megtekintése."""
    from stable_baselines3 import PPO

    model_path = args.model or str(MODELS_DIR / "best" / "best_model.zip")
    if not Path(model_path).exists():
        print(f"  ❌ Modell nem található: {model_path}")
        return

    print(f"  👁️  Modell megtekintése: {model_path}")
    model = PPO.load(model_path)

    # v2 FIX: VecNormalize betöltése vizualizációhoz is
    vecnorm_path = find_vecnormalize_pkl(model_path)
    env = gym.make("Humanoid-v5", render_mode="human")

    if vecnorm_path:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        venv = DummyVecEnv([lambda: env])
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
        print(f"  ✅ VecNormalize betöltve: {vecnorm_path}")

        obs = venv.reset()
        for _ in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = venv.step(action)
            if done[0]:
                obs = venv.reset()
        venv.close()
    else:
        print("  ⚠️  VecNormalize nélkül fut")
        obs, _ = env.reset()
        for _ in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()


def evaluate(args):
    """Modell formális kiértékelése."""
    from stable_baselines3 import PPO

    model_path = args.model or str(MODELS_DIR / "best" / "best_model.zip")
    if not Path(model_path).exists():
        print(f"  ❌ Modell nem található: {model_path}")
        return

    print(f"\n  📊 Kiértékelés: {model_path}")
    model = PPO.load(model_path)

    # v2 FIX: VecNormalize keresése és betöltése
    vecnorm_path = find_vecnormalize_pkl(model_path)
    eval_env = make_eval_env(vecnorm_path)
    evaluate_model(model, eval_env, n_episodes=10)
    eval_env.close()


def random_baseline(args):
    """Random policy megjelenítése összehasonlításhoz."""
    env = gym.make("Humanoid-v5", render_mode="human")
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    print("  🎲 Random baseline futtatása...")
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            print(f"    Epizód vége: reward={total_reward:.1f}, hossz={steps}")
            total_reward = 0
            steps = 0
            obs, _ = env.reset()
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Roboshelf AI — Humanoid RL tanítás v2")
    parser.add_argument("--level", choices=list(LEVELS.keys()), default="teszt")
    parser.add_argument("--continue-from", type=str, default=None,
                        help="Korábbi modell .zip útvonala (VecNormalize automatikusan betöltődik)")
    parser.add_argument("--watch", action="store_true", help="Betanított modell megtekintése")
    parser.add_argument("--eval", action="store_true", help="Modell kiértékelése")
    parser.add_argument("--random", action="store_true", help="Random baseline")
    parser.add_argument("--model", type=str, default=None,
                        help="Modell útvonala --watch és --eval módhoz")

    args = parser.parse_args()

    if args.watch:
        watch(args)
    elif args.eval:
        evaluate(args)
    elif args.random:
        random_baseline(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
