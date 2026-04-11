#!/usr/bin/env python3
"""
Roboshelf AI — Fázis 2: G1 lokomóció a retail boltban

Gymnasium wrapper a Unitree G1 navigációjához a kiskereskedelmi környezetben.
A robot megtanulja a boltban való járást: egyensúly, akadálykerülés, célnavigáció.

Feladat: A G1 a start pozícióból (y=0.5) eljut a raktárhoz (y=3.8).
Reward:
  - Előre haladás jutalma (y irányban)
  - Egyensúly megtartása (nem esik el)
  - Kontroll költség (simább mozgás)
  - Akadály-ütközés büntetés
  - Célba érkezés bonus

Observation: G1 propriocepció (qpos, qvel) + cél relatív pozíciója
Action: 29 DoF nyomaték parancsok

Használat:
  # Regisztrálás után:
  env = gym.make('RoboshelfRetailNav-v0')

  # Vagy közvetlenül:
  env = RoboshelfRetailNavEnv()
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import mujoco


# --- Környezet regisztráció ---
try:
    register(
        id='RoboshelfRetailNav-v0',
        entry_point='roboshelf_retail_nav_env:RoboshelfRetailNavEnv',
        max_episode_steps=1000,
    )
except gym.error.Error:
    pass  # Már regisztrálva


class RoboshelfRetailNavEnv(gym.Env):
    """
    Unitree G1 navigáció a Roboshelf retail bolt környezetben.

    A robot a bolt elején indul (y=0.5) és el kell jutnia a raktárhoz (y=3.8).
    Közben ki kell kerülnie a gondola polcokat és meg kell tartania az egyensúlyt.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # --- MJCF betöltés ---
        self._load_model()

        # --- Spaces ---
        # Observation: robot qpos (29 joint + 7 freejoint) + qvel (29+6) + target rel pos (2)
        obs_size = self._get_obs_size()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Action: 29 aktuátor (G1 joint position targets)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float64
        )

        # --- Feladat paraméterek ---
        self.target_pos = np.array([0.0, 3.8])  # Raktár pozíció (x, y)
        self.start_pos = np.array([0.0, 0.5])   # Robot start (x, y)

        # --- Reward súlyok (G1-re kalibrálva, 55 body) ---
        # FIX v4: "stand and fall" probléma végleges megoldása
        # Gyökér ok: w_healthy per-lépés bónusz → passzív "állási alapjövedelem" → PPO nem tanul mozogni
        # Megoldás (legged_gym + Gymnasium Humanoid-v4 közösségi konvenció):
        #   - w_healthy = 0.0 (teljesen ki! nem 0.5, hanem nulla)
        #   - w_forward az egyetlen pozitív reward forrás
        #   - w_fall mérsékelt: ne "fagyassza be" a robotot félelemből
        # v9: tracking reward + Humanoid-v4 mintájára kalibrált súlyok
        # forward_reward = w_forward * dot(velocity, direction_to_target)
        # Humanoid-v4 default: forward_reward_weight=1.25, healthy_reward=5.0
        # Mi: erős tracking, minimális healthy, mérsékelt fall penalty
        self.w_forward = 8.0       # Célkövetés — erős, domináns reward forrás
        self.w_healthy = 0.05      # Létezés bónusz — szinte nulla, nem jutalmaz passzív állást
        self.w_ctrl = -0.001       # Kontroll költség
        self.w_contact = -0.0001   # Kontakt költség
        self.w_goal = 100.0        # Célba érkezés bonus
        self.w_survival = 0.0      # Kikapcsolva
        self.w_fall = -20.0        # Mérsékelt esés büntetés
        self.w_gait = 0.0          # Kikapcsolva (curriculum: előbb járás stabilan, majd gait)

        # --- Gait paraméterek (ciklikus lépésminta) ---
        # 0.8s periódus = 1.25 lépés/mp, 50% offset = szimmetrikus bal-jobb váltás
        self._gait_period = 0.8    # másodperc
        self._gait_offset = 0.5   # jobb láb fázis offsetje (50% = szimmetrikus)
        self._stance_threshold = 0.55  # gait ciklus 55%-a stance, 45% swing
        self._episode_time = 0.0  # epizód eltelt ideje (másodperc)

        # --- Láb body indexek (betöltés után keressük meg) ---
        self._left_foot_id = None
        self._right_foot_id = None

        # --- Állapot ---
        self._healthy_z_range = (0.5, 1.5)  # G1 törzs magasság tartomány
        self._prev_dist_to_target = None

        # Renderer
        self._renderer = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, width=640, height=480)

    def _load_model(self):
        """Betölti a G1 + retail bolt kombinált MJCF modellt."""
        # Keressük a G1 modellt
        g1_candidates = [
            # Környezeti változó — Kaggle/Colab esetén ez az elsődleges
            os.environ.get("G1_MODEL_PATH"),
            # Kaggle klón (manuálisan klónozott menagerie)
            "/kaggle/working/mujoco_menagerie/unitree_g1",
            # MacBook Homebrew miniforge
            "/opt/homebrew/Caskroom/miniforge/base/lib/python3.13/site-packages/mujoco_playground/external_deps/mujoco_menagerie/unitree_g1",
            # pip-pel telepített mujoco_playground
            self._find_in_site_packages("mujoco_playground/external_deps/mujoco_menagerie/unitree_g1"),
            # Standalone menagerie
            self._find_in_site_packages("mujoco_menagerie/unitree_g1"),
            # Lokális klón
            os.path.expanduser("~/mujoco_menagerie/unitree_g1"),
        ]

        g1_dir = None
        for path in g1_candidates:
            if path and os.path.exists(os.path.join(str(path), "g1.xml")):
                g1_dir = str(path)
                break

        if g1_dir is None:
            # Fallback: MuJoCo Playground telepítése szükséges
            raise FileNotFoundError(
                "Unitree G1 MJCF nem található! Telepítsd: pip install playground\n"
                "Vagy klónozd: git clone https://github.com/google-deepmind/mujoco_menagerie.git"
            )

        # Retail bolt XML keresése
        store_xml = self._find_store_xml()

        # Kombinált XML létrehozása (a G1 mappából, hogy a meshdir helyes legyen)
        import tempfile
        combined_xml = f"""<mujoco model="roboshelf_g1_retail_nav">
  <include file="g1.xml"/>
  <include file="{store_xml}"/>
</mujoco>"""

        # Process-specifikus temp fájlnév: elkerüli a párhuzamos env-ek konfliktusát
        import tempfile
        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.xml', dir=g1_dir,
            prefix='_roboshelf_nav_', delete=False
        )
        tmp.write(combined_xml)
        tmp.close()
        self._combined_xml_path = tmp.name

        self.model = mujoco.MjModel.from_xml_path(self._combined_xml_path)
        self.data = mujoco.MjData(self.model)

        # G1 torso body index megkeresése
        self._torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self._torso_id == -1:
            # Fallback: első nem-world body
            self._torso_id = 1

        print(f"  ✅ Retail Nav env betöltve: {self.model.nbody} body, {self.model.nu} actuators")

        # Láb body indexek megkeresése (G1: ankle_roll_link)
        foot_candidates = [
            ("left_ankle_roll_link",  "right_ankle_roll_link"),   # G1
            ("left_ankle_link",       "right_ankle_link"),         # alternatív
            ("left_foot",             "right_foot"),               # generikus
        ]
        for left_name, right_name in foot_candidates:
            l = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, left_name)
            r = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, right_name)
            if l != -1 and r != -1:
                self._left_foot_id = l
                self._right_foot_id = r
                print(f"  ✅ Láb body-k: [{l}]{left_name}, [{r}]{right_name}")
                break
        if self._left_foot_id is None:
            print("  ⚠️  Láb body-k nem találhatók — contact pattern reward kikapcsolva")

    def _find_in_site_packages(self, subpath):
        """Keres egy útvonalat a site-packages-ben."""
        import sys
        for p in sys.path:
            if "site-packages" in p:
                full = os.path.join(p, subpath)
                if os.path.exists(full):
                    return full
        return None

    def _find_store_xml(self):
        """Megkeresi a retail bolt XML-t."""
        candidates = [
            os.path.join(os.path.dirname(__file__), "assets", "roboshelf_retail_store.xml"),
            os.path.join(os.getcwd(), "src", "envs", "assets", "roboshelf_retail_store.xml"),
            os.path.join(os.getcwd(), "roboshelf_retail_store.xml"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)

        raise FileNotFoundError(
            "roboshelf_retail_store.xml nem található!\n"
            f"Keresett helyek: {candidates}"
        )

    def _get_obs_size(self):
        """Kiszámítja az observation vektor méretét."""
        # qpos (nq) + qvel (nv) + target_rel_xy (2)
        return self.model.nq + self.model.nv + 2

    def _get_obs(self):
        """Összeállítja az observation vektort."""
        # Robot pozíció (x,y a padlón)
        robot_xy = self.data.body(self._torso_id).xpos[:2]
        target_rel = self.target_pos - robot_xy

        obs = np.concatenate([
            self.data.qpos.flat.copy(),
            self.data.qvel.flat.copy(),
            target_rel,
        ]).astype(np.float32)
        return obs

    def _is_healthy(self):
        """Ellenőrzi, hogy a robot áll-e (nem esett el)."""
        torso_z = self.data.body(self._torso_id).xpos[2]
        return self._healthy_z_range[0] < torso_z < self._healthy_z_range[1]

    def _compute_reward(self):
        """Kiszámítja a reward-ot."""
        robot_xy = self.data.body(self._torso_id).xpos[:2]
        dist_to_target = np.linalg.norm(self.target_pos - robot_xy)

        # 1. Célkövetés reward — sebesség × irány (Humanoid-v4 mintájára, de célirány-alapú)
        # A robot sebességének a cél irányába eső komponensét jutalmazzuk
        # Ez jobb mint y_velocity, mert a robot nem mindig y-irányban kell haladjon
        direction_to_target = self.target_pos - robot_xy
        dist_norm = np.linalg.norm(direction_to_target) + 1e-6
        direction_to_target = direction_to_target / dist_norm  # unit vector
        lin_vel = self.data.body(self._torso_id).cvel[3:5]  # cvel[3,4] = lin_x, lin_y
        forward_component = np.dot(lin_vel, direction_to_target)
        forward_reward = self.w_forward * forward_component
        self._prev_dist_to_target = dist_to_target

        # 2. Egyensúly jutalom
        healthy_reward = self.w_healthy if self._is_healthy() else 0.0

        # 3. Kontroll költség (simább mozgás = jobb)
        ctrl_cost = self.w_ctrl * np.sum(np.square(self.data.ctrl))

        # 4. Kontakt költség (ütközések büntetése)
        contact_forces = self.data.cfrc_ext.flat.copy()
        contact_cost = self.w_contact * np.sum(np.square(contact_forces))

        # 5. Cél bonus (ha elérte a raktárt)
        goal_bonus = 0.0
        if dist_to_target < 0.5:
            goal_bonus = self.w_goal

        # 6. Esés büntetés: csak termináláskor, egyszer
        fall_penalty = self.w_fall if not self._is_healthy() else 0.0

        # 7. Contact pattern reward (ciklikus gait timing)
        # A robot bal-jobb váltakozó lépésmintát tanul, nem csak egyensúlyoz
        gait_reward = 0.0
        if self._left_foot_id is not None:
            # Gait fázis kiszámítása az eltelt idő alapján
            phase = (self._episode_time % self._gait_period) / self._gait_period
            phase_left  = phase
            phase_right = (phase + self._gait_offset) % 1.0

            # Stance fázis: 0–55% = talaj érintés, 55–100% = swing (levegőben)
            left_should_contact  = phase_left  < self._stance_threshold
            right_should_contact = phase_right < self._stance_threshold

            # Kontakt detekció: Z-tengely kontakt erő > 1N
            left_contact  = self.data.cfrc_ext[self._left_foot_id,  2] > 1.0
            right_contact = self.data.cfrc_ext[self._right_foot_id, 2] > 1.0

            # XOR logika: reward ha kontakt egyezik a várt fázissal
            left_match  = not (left_contact  ^ left_should_contact)
            right_match = not (right_contact ^ right_should_contact)
            gait_reward = self.w_gait * (float(left_match) + float(right_match))

        total_reward = (forward_reward + healthy_reward + ctrl_cost + contact_cost
                        + goal_bonus + fall_penalty + gait_reward)

        info = {
            "forward_reward": forward_reward,
            "healthy_reward": healthy_reward,
            "ctrl_cost": ctrl_cost,
            "contact_cost": contact_cost,
            "goal_bonus": goal_bonus,
            "fall_penalty": fall_penalty,
            "gait_reward": gait_reward,
            "dist_to_target": dist_to_target,
            "torso_z": self.data.body(self._torso_id).xpos[2],
            "robot_xy": robot_xy.copy(),
        }

        return total_reward, info

    def reset(self, seed=None, options=None):
        """Környezet resetelése."""
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # G1 kezdő pozíció beállítása — a G1 XML "stand" keyframe alapján!
        # Forrás: mujoco_menagerie/unitree_g1/g1.xml <keyframe name="stand">
        # freejoint: [x, y, z, qw, qx, qy, qz]
        if self.model.nq > 7:
            self.data.qpos[0] = 0.0    # x
            self.data.qpos[1] = 0.5    # y (bolt eleje)
            self.data.qpos[2] = 0.79   # z — 0.79! (nem 0.75, az instabil volt)
            self.data.qpos[3] = 1.0    # qw
            self.data.qpos[4:7] = 0.0  # qx, qy, qz
            # Lábak: mind nulla (G1 egyenesen áll, nem hajlított térdekkel)
            self.data.qpos[7:19] = 0.0
            # Derék: nulla
            self.data.qpos[19:22] = 0.0
            # Bal kar: keyframe értékek (0.2 0.2 0 1.28 0 0 0)
            # Fontos: kar tömege megdönti a robotot ha nem a helyes pozícióban van!
            if self.model.nq >= 36:
                self.data.qpos[22] = 0.2
                self.data.qpos[23] = 0.2
                self.data.qpos[24] = 0.0
                self.data.qpos[25] = 1.28
                self.data.qpos[26:29] = 0.0
                # Jobb kar: (0.2 -0.2 0 1.28 0 0 0)
                self.data.qpos[29] = 0.2
                self.data.qpos[30] = -0.2
                self.data.qpos[31] = 0.0
                self.data.qpos[32] = 1.28
                self.data.qpos[33:36] = 0.0

        # ctrl alapértékek beállítása a keyframe alapján
        # Position control: ctrl = célpozíció. Nulla ctrl ≠ egyensúly!
        # A G1 keyframe ctrl értékei tartják fenn az egyensúlyi pozíciót.
        self._default_ctrl = np.zeros(self.model.nu)
        # Lábak (12 aktuátor, index 0-11): nulla
        # Derék (3 aktuátor, index 12-14): nulla
        # Bal kar (7 aktuátor, index 15-21): [0.2, 0.2, 0, 1.28, 0, 0, 0]
        # Jobb kar (7 aktuátor, index 22-28): [0.2, -0.2, 0, 1.28, 0, 0, 0]
        if self.model.nu >= 29:
            self._default_ctrl[15] = 0.2
            self._default_ctrl[16] = 0.2
            self._default_ctrl[17] = 0.0
            self._default_ctrl[18] = 1.28
            self._default_ctrl[22] = 0.2
            self._default_ctrl[23] = -0.2
            self._default_ctrl[24] = 0.0
            self._default_ctrl[25] = 1.28
        self.data.ctrl[:] = self._default_ctrl.copy()

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        self._prev_dist_to_target = np.linalg.norm(
            self.target_pos - self.data.body(self._torso_id).xpos[:2]
        )
        self._episode_time = 0.0  # Gait fázis időszámláló nullázása

        obs = self._get_obs()
        info = {"dist_to_target": self._prev_dist_to_target}

        return obs, info

    def step(self, action):
        """Egy szimuláció lépés végrehajtása."""
        # Akció skálázás: [-1, 1] → aktuátor tartomány
        action = np.clip(action, -1.0, 1.0)

        # Aktuátor vezérlés beállítása
        # Az akció a keyframe default_ctrl körüli offset (nem a tartomány közepe!)
        # Ezzel nulla akció = egyensúlyi pozíció, nem random ctrl_mean
        if self.model.nu > 0:
            ctrl_range = self.model.actuator_ctrlrange
            # Akció skála: 0.3 radian max eltérés az egyensúlyi pozíciótól
            # Teljes tartomány helyett kis perturbációk → stabilabb tanulás
            ACTION_SCALE = 0.3
            raw_ctrl = self._default_ctrl + action * ACTION_SCALE
            self.data.ctrl[:] = np.clip(raw_ctrl, ctrl_range[:, 0], ctrl_range[:, 1])

        # Fizikai szimuláció (2 sub-step — csökkentve 5-ről)
        # Kevesebb sub-step = lassabb fizika = robot tovább marad talpon = több tanulási jel
        for _ in range(2):
            mujoco.mj_step(self.model, self.data)

        # Gait időszámláló léptetése
        self._episode_time += 2 * self.model.opt.timestep

        # Observation
        obs = self._get_obs()

        # Reward
        reward, info = self._compute_reward()

        # Terminated: robot elesett
        terminated = not self._is_healthy()

        # Truncated: max lépésszám (Gymnasium kezeli)
        truncated = False

        return obs, reward, terminated, truncated, info

    def render(self):
        """Rendering."""
        if self.render_mode == "rgb_array" and self._renderer is not None:
            mujoco.mj_forward(self.model, self.data)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self):
        """Cleanup."""
        if self._renderer is not None:
            self._renderer.close()
        # Temp fájl törlése
        if hasattr(self, '_combined_xml_path') and os.path.exists(self._combined_xml_path):
            try:
                os.remove(self._combined_xml_path)
            except:
                pass


# === Standalone teszt ===
if __name__ == "__main__":
    print("Roboshelf Retail Nav Env — teszt")
    print("=" * 40)

    env = RoboshelfRetailNavEnv()
    obs, info = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space.shape}")
    print(f"Kezdő távolság a céltól: {info['dist_to_target']:.2f}m")
    print()

    # Random policy teszt
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            print(f"  Robot elesett a {step}. lépésnél")
            break

    print(f"\n  100 lépés reward: {total_reward:.1f}")
    print(f"  Végső távolság a céltól: {info['dist_to_target']:.2f}m")
    print(f"  Robot magasság: {info['torso_z']:.2f}m")
    print(f"\n✅ Env működik!")
    env.close()
