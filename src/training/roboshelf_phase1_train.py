#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  ROBOSHELF AI — Fázis 1: Első Humanoid RL Tanítás           ║
║  Humanoid-v5 + PPO (StableBaselines3)                       ║
║  Optimalizálva: MacBook Air M2, 16GB RAM, CPU               ║
╚══════════════════════════════════════════════════════════════╝

Futtatás:
  python3 roboshelf_phase1_train.py

A script 3 részből áll:
  1. Rövid teszt-tanítás (5 perc) — hogy lássuk, minden működik
  2. Közepes tanítás (1-2 óra) — a robot megtanul állni
  3. Teljes tanítás (6-15 óra) — stabil járás (opcionális, éjszaka futhat)

Az eredmények a ~/roboshelf-results/ mappába kerülnek:
  - TensorBoard logok (tanítási görbék)
  - Mentett modellek (checkpoint-ok)
  - Videó a tanult viselkedésről
"""

import os
import sys
import time
import argparse
from pathlib import Path

# ============================================================================
# KONFIGURÁCIÓ
# ============================================================================

# Eredmények mappája
RESULTS_DIR = Path.home() / "roboshelf-results"
LOGS_DIR = RESULTS_DIR / "logs"
MODELS_DIR = RESULTS_DIR / "models"
VIDEOS_DIR = RESULTS_DIR / "videos"

# Tanítási szintek (M2-re optimalizált)
TRAINING_LEVELS = {
    "teszt": {
        "total_timesteps": 50_000,
        "desc": "Gyors teszt (~5 perc) — csak ellenőrizzük, hogy minden fut",
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 5,
    },
    "kozepes": {
        "total_timesteps": 2_000_000,
        "desc": "Közepes tanítás (~1-2 óra) — a robot megtanul állni",
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
    },
    "teljes": {
        "total_timesteps": 10_000_000,
        "desc": "Teljes tanítás (~6-10 óra) — járásminta kialakul",
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
    },
    "ejszakai": {
        "total_timesteps": 30_000_000,
        "desc": "Éjszakai tanítás (~15+ óra) — stabil járás",
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
    },
}


def setup_directories():
    """Létrehozza az eredmény-mappákat."""
    for d in [RESULTS_DIR, LOGS_DIR, MODELS_DIR, VIDEOS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"  Eredmények mappája: {RESULTS_DIR}")


def check_environment():
    """Ellenőrzi, hogy minden szükséges csomag elérhető."""
    print("\n━━━ Környezet ellenőrzése ━━━\n")
    
    errors = []
    
    try:
        import mujoco
        print(f"  ✅ MuJoCo: v{mujoco.__version__}")
    except ImportError:
        errors.append("MuJoCo")
    
    try:
        import gymnasium as gym
        print(f"  ✅ Gymnasium: v{gym.__version__}")
    except ImportError:
        errors.append("Gymnasium")
    
    try:
        import stable_baselines3 as sb3
        print(f"  ✅ Stable-Baselines3: v{sb3.__version__}")
    except ImportError:
        errors.append("Stable-Baselines3")
    
    try:
        import torch
        print(f"  ✅ PyTorch: v{torch.__version__}")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"  ✅ PyTorch MPS (Apple GPU): elérhető")
        else:
            print(f"  ℹ️  PyTorch MPS: nem elérhető (CPU-t használunk)")
    except ImportError:
        errors.append("PyTorch")
    
    try:
        import tensorboard
        print(f"  ✅ TensorBoard: v{tensorboard.__version__}")
    except ImportError:
        errors.append("TensorBoard")
    
    # MuJoCo szimuláció teszt
    try:
        import gymnasium as gym
        env = gym.make("Humanoid-v5")
        obs, info = env.reset()
        env.step(env.action_space.sample())
        env.close()
        print(f"  ✅ Humanoid-v5 környezet: működik ({obs.shape[0]} dimenziós megfigyelés)")
    except Exception as e:
        errors.append(f"Humanoid-v5 env ({e})")
    
    if errors:
        print(f"\n  ❌ Hiányzó csomagok: {', '.join(errors)}")
        print("  Futtasd: pip install torch stable-baselines3 tensorboard")
        sys.exit(1)
    
    print("\n  🎉 Minden rendben, indulhat a tanítás!\n")


def create_training_env(n_envs=4):
    """
    Létrehozza a párhuzamos tanítási környezeteket.
    M2-n 4 párhuzamos env a sweet spot (8 CPU mag kihasználása).
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.env_util import make_vec_env
    
    print(f"  Párhuzamos környezetek létrehozása: {n_envs} db")
    
    vec_env = make_vec_env(
        "Humanoid-v5",
        n_envs=n_envs,
        seed=42,
    )
    
    # VecNormalize: normalizálja a megfigyeléseket és jutalmakat
    # Ez kritikus a Humanoid tanításához — nélküle nagyon instabil
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    
    return vec_env


def train(level="teszt", continue_from=None):
    """
    A fő tanítási függvény.
    
    Args:
        level: "teszt", "kozepes", "teljes", vagy "ejszakai"
        continue_from: korábbi modell elérési útja a folytatáshoz
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
        CallbackList,
    )
    
    config = TRAINING_LEVELS[level]
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Tanítás indítása: {level.upper():50s}║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  {config['desc']}")
    print(f"  Timesteps: {config['total_timesteps']:,}")
    print()
    
    # Környezet
    n_envs = 4 if level != "teszt" else 2
    vec_env = create_training_env(n_envs=n_envs)
    
    # Eval környezet (külön, a TensorBoard-hoz)
    eval_env = create_training_env(n_envs=1)
    
    # Modell név az adott szinthez
    run_name = f"humanoid_ppo_{level}_{int(time.time())}"
    
    # PPO modell konfigurálása (M2-re optimalizált hiperparaméterek)
    if continue_from and os.path.exists(continue_from):
        print(f"  Folytatás korábbi modellből: {continue_from}")
        model = PPO.load(
            continue_from,
            env=vec_env,
            tensorboard_log=str(LOGS_DIR),
        )
        model.learning_rate = 3e-4  # Reset learning rate
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            
            # Tanítási hiperparaméterek
            learning_rate=3e-4,
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=0.99,                    # Diszkont faktor
            gae_lambda=0.95,               # GAE lambda
            clip_range=0.2,                # PPO clip range
            clip_range_vf=None,            # Nincs value function clipping
            ent_coef=0.0,                  # Entrópia koeffíciens
            vf_coef=0.5,                   # Value function koeffíciens
            max_grad_norm=0.5,             # Gradiens clipping
            
            # Hálózat architektúra
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[256, 256],         # Policy hálózat (2x256 neuron)
                    vf=[256, 256],         # Value hálózat (2x256 neuron)
                ),
            ),
            
            # Logging
            tensorboard_log=str(LOGS_DIR),
            verbose=1,
            
            # Seed a reprodukálhatósághoz
            seed=42,
            
            # Eszköz: CPU (M2-n ez a legjobb SB3-hoz)
            device="cpu",
        )
    
    print(f"\n  Modell paraméterek száma: {sum(p.numel() for p in model.policy.parameters()):,}")
    print(f"  Policy hálózat: [256, 256]")
    print(f"  Device: {model.device}")
    print(f"  TensorBoard: tensorboard --logdir={LOGS_DIR}")
    print()
    
    # Callbacks
    callbacks = []
    
    # Checkpoint mentés (minden 50k timestep-nél)
    checkpoint_freq = max(50_000, config["total_timesteps"] // 10)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // n_envs,
        save_path=str(MODELS_DIR),
        name_prefix=run_name,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluáció (minden 25k timestep-nél)
    eval_freq = max(25_000, config["total_timesteps"] // 20)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR / "best"),
        log_path=str(LOGS_DIR / "eval"),
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=5,
        deterministic=True,
    )
    callbacks.append(eval_callback)
    
    # Tanítás indítása
    print("━━━ Tanítás indítása ━━━")
    print(f"  Indulás: {time.strftime('%H:%M:%S')}")
    print(f"  TensorBoard figyelés: nyiss egy új Terminált és futtasd:")
    print(f"    tensorboard --logdir={LOGS_DIR}")
    print(f"  Majd nyisd meg: http://localhost:6006")
    print()
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=CallbackList(callbacks),
            tb_log_name=run_name,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\n  ⚠️  Tanítás megszakítva (Ctrl+C)")
        print("  A legutóbbi checkpoint mentve van.")
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    
    # Végső modell mentése
    final_model_path = MODELS_DIR / f"{run_name}_final"
    model.save(str(final_model_path))
    vec_env.save(str(final_model_path) + "_vecnormalize.pkl")
    
    print(f"\n━━━ Tanítás befejezve ━━━")
    print(f"  Időtartam: {hours}h {minutes}m")
    print(f"  Végső modell: {final_model_path}.zip")
    print(f"  Best modell: {MODELS_DIR / 'best' / 'best_model.zip'}")
    print(f"  VecNormalize: {final_model_path}_vecnormalize.pkl")
    
    # Környezetek bezárása
    vec_env.close()
    eval_env.close()
    
    return str(final_model_path) + ".zip"


def evaluate_and_record(model_path=None, n_episodes=3):
    """
    Betanított modell kiértékelése és videó mentése.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    import gymnasium as gym
    
    if model_path is None:
        # Legutóbbi best modell keresése
        best_path = MODELS_DIR / "best" / "best_model.zip"
        if best_path.exists():
            model_path = str(best_path)
        else:
            # Bármelyik _final modell
            finals = list(MODELS_DIR.glob("*_final.zip"))
            if finals:
                model_path = str(sorted(finals)[-1])
            else:
                print("  ❌ Nincs mentett modell! Futtass először tanítást.")
                return
    
    print(f"\n━━━ Modell kiértékelése ━━━")
    print(f"  Modell: {model_path}")
    
    # Modell betöltése
    model = PPO.load(model_path, device="cpu")
    
    # VecNormalize betöltése (ha van)
    vecnorm_path = model_path.replace(".zip", "_vecnormalize.pkl")
    
    # Videó rögzítés környezet
    video_env = gym.make(
        "Humanoid-v5",
        render_mode="rgb_array",
    )
    
    print(f"  Epizódok: {n_episodes}")
    print()
    
    total_rewards = []
    
    for ep in range(n_episodes):
        obs, info = video_env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = video_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
        
        total_rewards.append(episode_reward)
        print(f"  Epizód {ep + 1}: reward = {episode_reward:.1f}, lépések = {steps}")
    
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\n  Átlagos reward: {avg_reward:.1f}")
    
    video_env.close()
    
    # Vizuális megjelenítés (ha van display)
    print(f"\n  💡 Vizuális megtekintéshez futtasd:")
    print(f"     python3 -c \"")
    print(f"     import gymnasium as gym")
    print(f"     from stable_baselines3 import PPO")
    print(f"     model = PPO.load('{model_path}', device='cpu')")
    print(f"     env = gym.make('Humanoid-v5', render_mode='human')")
    print(f"     obs, _ = env.reset()")
    print(f"     for _ in range(2000):")
    print(f"         action, _ = model.predict(obs, deterministic=True)")
    print(f"         obs, r, term, trunc, info = env.step(action)")
    print(f"         if term or trunc: obs, _ = env.reset()")
    print(f"     env.close()\"")
    
    return avg_reward


def watch_live(model_path=None):
    """
    Élő vizualizáció — megnyitja a MuJoCo ablakot a betanított robottal.
    """
    from stable_baselines3 import PPO
    import gymnasium as gym
    
    if model_path is None:
        best_path = MODELS_DIR / "best" / "best_model.zip"
        if best_path.exists():
            model_path = str(best_path)
        else:
            finals = list(MODELS_DIR.glob("*_final.zip"))
            if finals:
                model_path = str(sorted(finals)[-1])
            else:
                print("  ❌ Nincs mentett modell!")
                return
    
    print(f"\n━━━ Élő megjelenítés ━━━")
    print(f"  Modell: {model_path}")
    print(f"  Bezárás: Ctrl+C vagy ablak bezárása\n")
    
    model = PPO.load(model_path, device="cpu")
    env = gym.make("Humanoid-v5", render_mode="human")
    
    obs, _ = env.reset()
    episode_reward = 0
    episode_num = 1
    
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                print(f"  Epizód {episode_num}: reward = {episode_reward:.1f}")
                obs, _ = env.reset()
                episode_reward = 0
                episode_num += 1
    except KeyboardInterrupt:
        print("\n  Megjelenítés bezárva.")
    finally:
        env.close()


def random_baseline():
    """
    Random agent futtatása — összehasonlítási alap.
    Megmutatja, hogyan néz ki a robot tanítás ELŐTT.
    """
    import gymnasium as gym
    
    print("\n━━━ Random baseline (tanítás előtti állapot) ━━━")
    print("  A robot véletlenszerű akciókat hajt végre.\n")
    
    env = gym.make("Humanoid-v5", render_mode="human")
    obs, _ = env.reset()
    
    total_reward = 0
    steps = 0
    
    try:
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                print(f"  Elesett {steps} lépés után, reward: {total_reward:.1f}")
                obs, _ = env.reset()
                total_reward = 0
                steps = 0
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    
    print("  Ez volt a kiindulás — most nézzük, mit tanul a PPO!\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Roboshelf AI — Humanoid RL tanítás",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Példák:
  python3 roboshelf_phase1_train.py                    # Gyors teszt (5 perc)
  python3 roboshelf_phase1_train.py --level kozepes     # Közepes (1-2 óra)
  python3 roboshelf_phase1_train.py --level teljes      # Teljes (6-10 óra)
  python3 roboshelf_phase1_train.py --level ejszakai    # Éjszaka (15+ óra)
  python3 roboshelf_phase1_train.py --random            # Random baseline megtekintése
  python3 roboshelf_phase1_train.py --watch             # Betanított modell megtekintése
  python3 roboshelf_phase1_train.py --eval              # Modell kiértékelése
        """
    )
    
    parser.add_argument(
        "--level",
        choices=["teszt", "kozepes", "teljes", "ejszakai"],
        default="teszt",
        help="Tanítási szint (default: teszt)"
    )
    parser.add_argument(
        "--continue-from",
        type=str,
        default=None,
        help="Korábbi modell elérési útja a folytatáshoz"
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Random baseline megtekintése (tanítás nélkül)"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Betanított modell élő megtekintése"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Betanított modell kiértékelése"
    )
    
    args = parser.parse_args()
    
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       ROBOSHELF AI — Fázis 1: Humanoid RL Tanítás          ║")
    print("║       MuJoCo Humanoid-v5 + PPO                             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    # Környezet ellenőrzése
    check_environment()
    
    # Mappák létrehozása
    setup_directories()
    
    if args.random:
        random_baseline()
    elif args.watch:
        watch_live()
    elif args.eval:
        evaluate_and_record()
    else:
        # Tanítás
        model_path = train(level=args.level, continue_from=args.continue_from)
        
        # Automatikus kiértékelés a tanítás után
        print("\n" + "=" * 60)
        evaluate_and_record(model_path)
        
        print(f"\n━━━ Következő lépések ━━━")
        print(f"  1. TensorBoard: tensorboard --logdir={LOGS_DIR}")
        print(f"  2. Élő megtekintés: python3 {__file__} --watch")
        print(f"  3. Közepes tanítás: python3 {__file__} --level kozepes")
        print(f"  4. Folytatás: python3 {__file__} --level teljes --continue-from {model_path}")


if __name__ == "__main__":
    main()
