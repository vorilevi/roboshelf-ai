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

        # --- Reward súlyok (v14: epizód végi dist bonus — az igazi navigációs jel) ---
        # v13 tanulság: sub-step=1 → cvel más skálán → tracking negatív → összeomlik
        #               sub-step=2 visszaállítva
        # v12b tanulság: per-lépés dist_shaping GYENGE (0.002m/lép × w_dist = semmi)
        #               Math: 0.5 m/s × 0.004s = 0.002m/lép → még w_dist=500-zal is 1/lép
        #
        # MEGOLDÁS: epizód VÉGI egyszeri dist bonus (termináláskor vagy truncation-kor)
        #   w_dist_final=200 × (start_dist - final_dist):
        #   - Ha 86 lépés alatt 0.5m-t megy: +200×0.5 = +100 bonus (= 1.16/lépés ekvivalens)
        #   - Ez ERŐSEBB mint a tracking reward → DOMINÁL a navigáció
        #   - Per-lépés dist shaping lecsökkentve (vestigial, stabilizáló hatás)
        #
        # Kulcsszámok:
        #   - Ha robot 3.3m-nél marad: +0 dist bonus
        #   - Ha 2.5m-re megy (0.8m): +200×0.8 = +160 bonus az epizód végén
        #   - Ha célba ér (0m): +200×3.3 = +660 → erős végső motiváció
        self.w_forward = 8.0         # Sebesség × célirány — VÁLTOZATLAN (v11 bevált érték)
        self.w_dist = 2.0            # Per-lépés potential shaping (kis súly, csak stabilizál)
        self.w_dist_final = 200.0    # Epizód VÉGI egyszeri dist bonus [ÚJ v14!]
        self.w_proximity = 2.0       # Proximity bonus 3.5m küszöbön — marad
        self.w_air_time = 1.0        # Feet air time jutalom [ÚJ v14!]
                                     # legged_gym/Isaac Gym konvenció: valódi lépést jutalmaz
                                     # cfrc_ext[foot,2]<=0 → levegőben → +reward
                                     # Megakadályozza a "shuffle" (csoszogó) mozgásmintát
        self.w_healthy = 0.05        # Minimális alive bonus (MÁR 100×-os csökkentés v4-hez képest)
                                     # Annealing NEM szükséges: 0.05×86=4.15 = 2.7% a total rewardnak
        self.w_ctrl = -0.001         # Kontroll költség — VÁLTOZATLAN
        self.w_contact = -0.0001     # Kontakt költség — VÁLTOZATLAN
        self.w_goal = 100.0          # Célba érkezés bonus — VÁLTOZATLAN
        self.w_fall = -20.0          # Esés büntetés — VÁLTOZATLAN
        self.w_gait = 0.0            # Kikapcsolva (curriculum)

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
        """Kiszámítja a reward-ot (v12: navigation-centrikus shaping)."""
        robot_xy = self.data.body(self._torso_id).xpos[:2]
        dist_to_target = np.linalg.norm(self.target_pos - robot_xy)

        # 1. Sebesség-alapú tracking reward (v11-ből marad, kisebb súllyal)
        # A robot sebességének a cél irányába eső komponensét jutalmazzuk
        direction_to_target = self.target_pos - robot_xy
        dist_norm = np.linalg.norm(direction_to_target) + 1e-6
        dir_unit = direction_to_target / dist_norm  # unit vector
        lin_vel = self.data.body(self._torso_id).cvel[3:5]  # cvel[3,4] = lin_x, lin_y
        forward_component = np.dot(lin_vel, dir_unit)
        forward_reward = self.w_forward * forward_component

        # 2. Potential-based distance shaping (ÚJ v12)
        # Ng et al. 1999: F(s,s') = γ·Φ(s') - Φ(s), ahol Φ(s) = -dist (közelebb = nagyobb potenciál)
        # Egyszerűsítve: (prev_dist - curr_dist) × w_dist
        # Garantálja: közeledés → pozitív, távolodás → negatív, reset-en nulla
        # Nem torzítja az optimális policy-t (ellentétben a direkt dist penalty-vel)
        dist_delta = (self._prev_dist_to_target - dist_to_target)  # pozitív ha közelebb
        dist_shaping = self.w_dist * dist_delta
        self._prev_dist_to_target = dist_to_target

        # 3. Proximity bonus (v12b: küszöb 3.5m → azonnal aktív!)
        # v12 hiba: 2.0m küszöb soha nem aktiválódott (robot 3.22m-en maradt)
        # v12b javítás: 3.5m küszöb → az epizód ELSŐ lépésétől aktív jutalom
        # Start: 3.3m → kis pozitív bonus. Cél (0m): w_proximity
        # Lineáris skálázás: minél közelebb, annál több
        PROXIMITY_THRESHOLD = 3.5  # méter — start=3.3m tehát AZONNAL aktív
        proximity_bonus = 0.0
        if dist_to_target < PROXIMITY_THRESHOLD:
            # Lineáris: 3.5m → 0.0, 0.0m → w_proximity
            proximity_bonus = self.w_proximity * (PROXIMITY_THRESHOLD - dist_to_target) / PROXIMITY_THRESHOLD

        # 4. Egyensúly jutalom
        healthy_reward = self.w_healthy if self._is_healthy() else 0.0

        # 5. Kontroll költség — az AKCIÓRA számítva (action²)
        ctrl_cost = self.w_ctrl * np.sum(np.square(self._last_action))

        # 6. Kontakt költség (ütközések büntetése)
        contact_forces = self.data.cfrc_ext.flat.copy()
        contact_cost = self.w_contact * np.sum(np.square(contact_forces))

        # 7. Cél bonus (ha elérte a raktárt, 0.5m-en belül)
        goal_bonus = 0.0
        if dist_to_target < 0.5:
            goal_bonus = self.w_goal

        # 8. Esés büntetés: csak termináláskor, egyszer
        fall_penalty = self.w_fall if not self._is_healthy() else 0.0

        # 9. Feet air time jutalom (ÚJ v14 — legged_gym/Isaac Gym konvenció)
        # A robot valódi lépésre kénytelen: ha mindkét láb a talajon → 0 reward
        # Ha legalább egy láb levegőben → +reward
        # Megakadályozza a "shuffle" (csoszogó, talpak nem emelkednek) mozgásmintát
        # cfrc_ext[body_id, 2] = z-irányú kontakt erő (>1N → talaj érintés)
        air_time_reward = 0.0
        if self._left_foot_id is not None and self.w_air_time > 0.0:
            left_contact  = self.data.cfrc_ext[self._left_foot_id,  2] > 1.0
            right_contact = self.data.cfrc_ext[self._right_foot_id, 2] > 1.0
            # Jutalom: legalább egy láb levegőben ÉS a robot mozog (tracking reward pozitív)
            # Ha álló/shuffle: mindkét láb talajon → 0
            # Ha lépked: váltakozva levegőben → +reward
            n_air = (not left_contact) + (not right_contact)  # 0, 1 vagy 2
            # Csak akkor jutalmaz, ha a robot tényleg halad (tracking pozitív)
            # Ez megakadályozza hogy a robot "levegőbe rúgjon" állva
            if forward_component > 0:
                air_time_reward = self.w_air_time * n_air

        # 10. Régi gait timing reward (fázis-alapú) — kikapcsolva, curriculum-ra vár
        gait_reward = 0.0
        if self._left_foot_id is not None and self.w_gait > 0.0:
            phase = (self._episode_time % self._gait_period) / self._gait_period
            phase_left  = phase
            phase_right = (phase + self._gait_offset) % 1.0
            left_should_contact  = phase_left  < self._stance_threshold
            right_should_contact = phase_right < self._stance_threshold
            left_contact  = self.data.cfrc_ext[self._left_foot_id,  2] > 1.0
            right_contact = self.data.cfrc_ext[self._right_foot_id, 2] > 1.0
            left_match  = not (left_contact  ^ left_should_contact)
            right_match = not (right_contact ^ right_should_contact)
            gait_reward = self.w_gait * (float(left_match) + float(right_match))

        total_reward = (forward_reward + dist_shaping + proximity_bonus
                        + healthy_reward + ctrl_cost + contact_cost
                        + goal_bonus + fall_penalty + air_time_reward + gait_reward)

        info = {
            "forward_reward": forward_reward,
            "dist_shaping": dist_shaping,
            "proximity_bonus": proximity_bonus,
            "healthy_reward": healthy_reward,
            "ctrl_cost": ctrl_cost,
            "contact_cost": contact_cost,
            "goal_bonus": goal_bonus,
            "fall_penalty": fall_penalty,
            "air_time_reward": air_time_reward,
            "gait_reward": gait_reward,
            "dist_to_target": dist_to_target,
            "dist_delta": dist_delta,
            "torso_z": self.data.body(self._torso_id).xpos[2],
            "robot_xy": robot_xy.copy(),
        }

        return total_reward, info

    def _compute_final_dist_bonus(self, dist_to_target):
        """Epizód végi egyszeri dist bonus (v14).

        A per-lépés dist_shaping túl gyenge (0.002m/lép → semmi).
        Epizód végén egyszer: w_dist_final × (start_dist - final_dist)
        Ha robot 0.5m-t ment: +100 bonus → erősebb mint az összes tracking reward.
        """
        dist_improvement = self._start_dist_to_target - dist_to_target
        return self.w_dist_final * max(dist_improvement, 0.0)  # csak közeledésért!

    def reset(self, seed=None, options=None):
        """Környezet resetelése."""
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # G1 kezdő pozíció beállítása — a G1 XML "stand" keyframe alapján!
        # Forrás: mujoco_menagerie/unitree_g1/g1.xml <keyframe name="stand">
        # freejoint: [x, y, z, qw, qx, qy, qz]
        # Reset zaj: Humanoid-v4 mintájára (reset_noise_scale=1e-2)
        # Nélküle: determinisztikus reset → policy mindig ugyanazt csinálja (±0 szórás eval-ban)
        noise_scale = 0.01
        noise = self.np_random.uniform(-noise_scale, noise_scale, self.model.nq)

        if self.model.nq > 7:
            self.data.qpos[0] = 0.0 + noise[0]   # x
            self.data.qpos[1] = 0.5 + noise[1]   # y (bolt eleje)
            self.data.qpos[2] = 0.79              # z — nem zajos, maradjon stabil magasság
            self.data.qpos[3] = 1.0               # qw — nem zajos
            self.data.qpos[4:7] = noise[4:7] * 0.001  # kis quaternion zaj
            # Lábak: nulla + kis zaj
            self.data.qpos[7:19] = noise[7:19]
            # Derék: nulla + kis zaj
            self.data.qpos[19:22] = noise[19:22]
            # Karok: keyframe értékek + kis zaj
            if self.model.nq >= 36:
                self.data.qpos[22] = 0.2  + noise[22]
                self.data.qpos[23] = 0.2  + noise[23]
                self.data.qpos[24] = 0.0  + noise[24]
                self.data.qpos[25] = 1.28 + noise[25]
                self.data.qpos[26:29] = noise[26:29]
                self.data.qpos[29] = 0.2  + noise[29]
                self.data.qpos[30] = -0.2 + noise[30]
                self.data.qpos[31] = 0.0  + noise[31]
                self.data.qpos[32] = 1.28 + noise[32]
                self.data.qpos[33:36] = noise[33:36]

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
        self._start_dist_to_target = self._prev_dist_to_target  # v14: epizód végi dist bonus
        self._episode_time = 0.0  # Gait fázis időszámláló nullázása
        self._last_action = np.zeros(self.model.nu)  # ctrl cost inicializálás

        obs = self._get_obs()
        info = {"dist_to_target": self._prev_dist_to_target}

        return obs, info

    def step(self, action):
        """Egy szimuláció lépés végrehajtása."""
        # Akció skálázás: [-1, 1] → aktuátor tartomány
        action = np.clip(action, -1.0, 1.0)
        self._last_action = action.copy()  # ctrl cost számításhoz (Humanoid-v4: action² alapú)

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

        # Fizikai szimuláció (2 sub-step)
        # v13 tanulság: sub-step=1 → tracking reward negatív lesz (cvel kisebb/más irányú)
        # A v11-es policy 2 sub-step fizikán tanult és jól működött (+133 reward)
        # → Visszaállítás 2 sub-stepre
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

        # v14: Epizód végi dist bonus — termináláskor EGYSZER hozzáadjuk
        # Ez a per-lépés dist_shaping hiányát pótolja (matematikailag domináns jel)
        # truncated esetén is adjuk (max_episode_steps elért = sikeres hosszú ep)
        if terminated or truncated:
            dist_to_target = info["dist_to_target"]
            final_bonus = self._compute_final_dist_bonus(dist_to_target)
            reward += final_bonus
            info["final_dist_bonus"] = final_bonus
            info["dist_improvement"] = self._start_dist_to_target - dist_to_target
        else:
            info["final_dist_bonus"] = 0.0
            info["dist_improvement"] = 0.0

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
