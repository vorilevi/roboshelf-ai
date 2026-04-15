# Roboshelf AI — Session Kontextus

> Ezt a fájlt az AI olvassa, hogy gyorsan felvegye a fonalat.
> Utoljára frissítve: 2026-04-15 (v18 env kész: smoothness penalties + feltételes air_time)

---

## Fontos: munkakörnyezet és terminal parancsok

**Repo helye Mac-en:** `~/roboshelf-ai-dev/roboshelf-ai/`
**Python környezet:** miniforge/conda, a terminálban aktiválva van (nem kell külön aktiválni)

**Tanítás indítása (fresh start):**
```bash
cd ~/roboshelf-ai-dev/roboshelf-ai
python src/training/roboshelf_phase2_train.py --level m2_3m_nogait
```

**Fine-tune indítása (meglévő modellből folytatás):**
```bash
cd ~/roboshelf-ai-dev/roboshelf-ai
python src/training/roboshelf_phase2_finetune.py --steps 6000000 --lr 1e-4 --clip 0.15
```

**TensorBoard:**
```bash
cd ~/roboshelf-ai-dev/roboshelf-ai
tensorboard --logdir roboshelf-results/phase2/logs
# → http://localhost:6006
```

**GitHub push:**
```bash
cd ~/roboshelf-ai-dev/roboshelf-ai
git add -A && git commit -m "..." && git push
```

**FONTOS:** Az AI sandbox-ból (Cowork) NEM tud git push-olni (nincs auth) — mindig Mac terminálból kell!

---

## Mi ez a projekt?

Humanoid robot RL tanítás: Unitree G1 robot megtanul egy kiskereskedelmi boltban
polcokat feltölteni. Végcél: befektetői demo. 5 fázis.

**Stack:** MuJoCo 3.6+, Stable-Baselines3 PPO, PyTorch, MacBook M2 (dev) + Kaggle T4 (tanítás)

---

## Fázisok állapota

| Fázis | Státusz | Leírás |
|-------|---------|--------|
| 1 | ✅ Kész | Humanoid-v5 lokomóció (referencia tanítás) |
| 2 | 🔄 Tanítás fut | G1 retail navigáció (start→raktár, 3.3m) |
| 3 | 📝 Vázlat kész | Pick & place manipuláció (env megírva, tanítás még nem) |
| 4 | ⬜ Tervezett | Hierarchikus policy |
| 5 | ⬜ Tervezett | Befektetői demo |

---

## Fázis 2 — részletek

**Feladat:** G1 robot a bolt elejéről (y=0.5) eljut a raktárhoz (y=3.8), 3.3m távolság.

**Env:** `src/envs/roboshelf_retail_nav_env.py`
- Obs: qpos+qvel (G1, 29 DoF) + target relatív pozíció = 346 dim, float32
- Action: 29 DoF nyomaték, [-1, 1]
- Reward: előre haladás + egyensúly (w=5) + kontroll cost + cél bónusz (100)
- Reward súlyok G1-re kalibrálva: w_ctrl=-0.001, w_contact=-0.0001

**Tanítóscript:** `src/training/roboshelf_phase2_train.py`
- Szintek: teszt / kozepes / teljes / m2_2ora
- Device: CPU (MPS ki van zárva — float64 konfliktus)
- Fontos fix: float32 obs dtype, CPU device, VecNormalize sync callback

**Baseline eredmény (100k teszt, MacBook):**
- Átlag reward: -5653.9, ep hossz: 43 lépés, cél táv: 3.28m
- Értelmezés: robot 43 lépés után esik el, alig mozdul — ez normális 100k-nál

---

## Ami éppen fut

**KÉSZ** — v18 env + m2_20m_v18 szint megírva (2026-04-15):
- `src/envs/roboshelf_retail_nav_env.py` → v18: feltételes air_time + smoothness penalties
- `src/training/roboshelf_phase2_train.py` → `m2_20m_v18` + `ent_coef` per-szint konfig

**v17 20M eredménye (2026-04-15):**
- `python src/training/roboshelf_phase2_train.py --level m2_20m_v17`
- 8M-nál: reward=+199 (curriculum működött!) → 12M-nál: -15241 (kapálózás beégett)
- Diagnózis: feltétel nélküli air_time → kapálózás optimum; 6M fázis1 túl hosszú → entrópia elfogyott

---

## Következő teendők (prioritás sorrendben)

1. **GitHub push + v18 fresh start (20M)** — Mac terminálból:
   ```bash
   cd ~/roboshelf-ai-dev/roboshelf-ai
   git add -A && git commit -m "v18: smoothness penalties + feltételes air_time + ent_coef=0.01" && git push
   python src/training/roboshelf_phase2_train.py --level m2_20m_v18
   ```
2. **Fázis 3 tanítóscript megírása** — `src/envs/roboshelf_manipulation_env.py` már megvan,
   csak a `src/training/roboshelf_phase3_train.py` hiányzik

---

## Fontos fájlok

```
src/envs/roboshelf_retail_nav_env.py       ← Fázis 2 env (v11: reset noise, tracking reward, w_healthy=0.05)
src/envs/roboshelf_manipulation_env.py     ← Fázis 3 env (vázlat, kész)
src/envs/assets/roboshelf_retail_store.xml ← Bolt MJCF (2 gondola, termékek)
src/training/roboshelf_phase2_train.py     ← Fázis 2 tanítás (m2_10m_v11 szint az utolsó)
src/training/roboshelf_phase2_finetune.py  ← Fine-tune script (evaluations.npz append-del)
src/training/humanoid_v4_baseline.py       ← Humanoid-v4 baseline (3M → reward=855 ✅)
notebooks/roboshelf_phase2_kaggle.ipynb    ← Kaggle notebook (KaggleFlushCb + SyncNormCb fix)
src/roboshelf_phase2_check.py              ← Rendszerellenőrző
roboshelf-results/phase2/models/best/      ← Legjobb modell (20M lépés, reward=+133.6, ep=83)
roboshelf-results/phase2/logs/             ← TensorBoard logok + evaluations.npz (teljes história)
```

---

## Tanítási előzmények (Fázis 2)

| Lépés | Átlag reward | Ep hossz | Megjegyzés |
|-------|-------------|----------|------------|
| 100k  | -5653.9     | 43       | baseline teszt |
| 600k  | -216.6      | 39       | m2_2ora |
| 1.2M  | -81.6       | 40       | m2_2ora |
| 1.8M  | -53.5       | 42       | m2_2ora |
| 2.4M  | -52.1       | 43       | m2_2ora, plató |
| 3.0M  | -41.4       | 43       | m2_2ora final |
| 4.2M  | -31.5       | 45       | finetune 1 (+survival bónusz) |
| 5.4M  | -27.4       | 46       | finetune 1 |
| 6.6M  | +24.2       | 47       | finetune 2 ← **első pozitív!** |
| 7.8M  | +42.0       | 50       | finetune 2 ← **ep plató áttörve!** |
| 10.2M | +51.3       | 51       | finetune 2 final — **legjobb MacBook modell** |
| 12.2M | -84.2       | 51       | finetune 3 — contact pattern sokk |
| 14.2M | -81.4       | 52       | finetune 3 — lassú javulás |
| 16.2M | -81.9       | 52       | finetune 3 — plató |
| 18.2M | -81.8       | 52       | finetune 3 — plató → **fresh start Kaggle-n** |
| 3.0M  | -130.4      | 41       | m2_3m_fresh (w_gait=0.18) — gait zavarja a korai tanulást |
| 7.8M  | -113.3      | 43       | m2_3m_fresh finetune (+6M) — 43 ep hossznál plató, gait kikapcsolva |
| 3.0M  | -130.4      | 41       | m2_3m_nogait — **stand-and-fall**: minden ep azonos (-135, 41 lépés, ±0) |
| 3.0M  | -264.6      | 35       | m2_3m_v3 — w_fall=-50 túl erős → robot "befagyott" |
| 3.0M  | -249.3      | 32       | m2_3m_v4 — w_healthy=0.0 + rossz pozíció → azonnal elesett |
| 3.0M  | -4.12       | 28-30    | m2_3m_v5 — helyes z=0.79, de ctrl skálázás hibás → süllyed |
| 3.0M  | -2.50       | 30       | m2_3m_v6 — akció fix OK, de 5 sub-step túl gyors → 30 lépésnél süllyed |
| 3.0M  | -264.6      | 67       | m2_3m_v7 — sub-step 2, stabil alap, de policy előre dől → 67 lépésnél terminál |
| 3.0M  | -130.4      | 79       | m2_3m_v8 — ACTION_SCALE=0.3, ±0.0 std (determinisztikus reset még!) |
| 3.0M  | -2.50       | 79       | m2_5m_v9 — tracking reward, w_healthy=0.05, ±0.0 std (még mindig!) |
| 10.0M | +78.7       | 79       | m2_10m_v11 — **ÁTTÖRÉS: reset noise_scale=0.01! ±29.4 std!** |
| 20.0M | +133.6      | 83       | finetune 10M (lr=5e-5, clip=0.1) — eval görbe még emelkedik |
| ~21M  | -121.3      | 83       | v12 finetune — catastrophic forgetting! (w_forward 8→4 scale shift) |
| ~22M  | +94.8       | 86       | v12b finetune — visszaállt, de 86 lépéses fizikai határ megmarad |
| 10M   | -330.9      | 169      | v13 fresh (sub-step=1) — cvel skála megváltozott → tracking negatív |
| 10M   | -44.1       | 85       | v14 fresh (ep-végi dist bonus + air_time) — eval görbe: 2M→-302, 10M→-45 (emelkedik!) |
| 16M   | +3.7        | 85       | v14 finetune (+6M, lr=5e-5) — **lokális optimum: robot áll, táv=3.18m (12cm haladás!)** |
| 10M   | -244.9      | 75       | v15 fresh — stuck minden ep 75 lépésnél (w_stuck gyenge + air_time gyenge) |
| 10M   | -192.7      | 40       | v16 fresh — stuck minden ep 40 lépésnél (PPO ablak méretét tanulta!) |

**KRITIKUS ÁTTÖRÉS (v11):** A reset noise_scale=0.01 bevezetése törte át a determinisztikus ±0.0 std falat. A policy most általánosít, nem ragad lokális optimumba.
**Áttörés (korábbi):** 7.8M lépésnél a reward pozitívba fordult (+42) és az ep hossz áttörte a 43 lépéses plafont (50 lépés).
**Contact pattern bevezetése (12.2M+):** visszaesés -84-re, majd plató 52 ep hossznál. A gait reward (w=0.18) túl gyenge volt a régi modell "szokásaival" szemben → fresh start szükséges.
**Fresh start (m2_3m_fresh, w_gait=0.18):** 3M+6M lépés, de 43 ep hossznál megragadt. A gait reward konfliktusban volt a korai járásminta kialakításával.
**Tanulság:** Gait reward-ot csak akkor szabad bevezetni, ha a robot már tud járni (reward > 0, ep hossz > 100). Curriculum megközelítés szükséges.
**Jelenlegi legjobb modell:** 20M lépés összesen, reward=+133.6, ep hossz=83 (`roboshelf-results/phase2/models/best/best_model.zip`)

---

## Ismert bugok / döntések

- MPS (Apple Silicon GPU) ki van zárva: SB3 MlpPolicy float64-et használ, MPS nem támogatja
- `find /` timeout Kaggle-n: ezért a G1 modellt direktben klónozzuk (`git clone mujoco_menagerie`)
- G1_MODEL_PATH env változó: a nav env ezt nézi először, utána fallback útvonalak
- contact_cost csökkentve (G1-nek 55 bodyja van, Humanoid-v5-nek csak 13)
- evaluations.npz: finetune futtatás előtt backup készül (`evaluations_before_finetune.npz`), majd merge történik — nem vész el a korábbi história
- roboshelf-results mappa: áthelyezve `~/Documents/`-ből a repo gyökerébe (`roboshelf-ai/roboshelf-results/`)
- Kaggle IOPub timeout fix: `KaggleFlushCb` — 200 lépésenként flush, 2000 lépésenként heartbeat, `progress_bar=False`
- Kaggle `kozepes` futás timeout miatt csak 4M lépésig jutott (best model reward: -60.3) — nem használjuk
- SyncNormCb: csak 10 lépésenként szinkronizál
- **Stand-and-fall probléma** (jól ismert RL hiba): w_healthy=3.0 per-lépés bónusz → robot megtanul állni és elesni. Fix: w_healthy=1.0 (mérsékelt), w_forward=5.0 (domináns), w_fall=-20.
- **G1 kezdőpozíció hiba (KRITIKUS)**: reset-kor z=0.75 + karok nulla szögben → fizikailag instabil, azonnal elesett. Fix: G1 XML keyframe ("stand") alapján z=0.79, kar joint szögek: bal=[0.2, 0.2, 0, 1.28, 0, 0, 0], jobb=[0.2, -0.2, 0, 1.28, 0, 0, 0].
- **Akció skálázás hiba (KRITIKUS)**: `ctrl = ctrl_mean + action * ctrl_half` — a ctrl_mean a ctrlrange közepe, ami nem az egyensúlyi pozíció! Fix: `ctrl = default_ctrl + action * ctrl_half`, ahol default_ctrl a keyframe ctrl értékei. Így nulla akció = egyensúlyi pozíció tartása.
- **Sub-step hiba**: 5 sub-step/lépés túl gyors fizika → robot 30 lépés alatt összecsuszik még helyes ctrl-lel is. Fix: 2 sub-step → robot 100+ lépésen át stabil nulla akcióval (tesztelve ✅).
- Contact pattern reward: w_gait=0.18 túl gyenge volt fine-tune-ban (plató 52 ep hossznál) → fresh start-ban erősebb gradienst ad (nulláról tanulja meg egyszerre a járást és a gait timing-ot)
- Kaggle T4 GPU: MuJoCo csak CPU-n fut → GPU teljesen kihasználatlan. SB3 MlpPolicy CPU-ra van optimalizálva (issue #1245). Multi-GPU nem támogatott SB3-ban. Következtetés: Kaggle T4x2 csomag felesleges a mi feladatunkhoz.
- Optimális Kaggle konfig (ha újra kellene): device='cpu', n_envs=4 (= 4 CPU core), batch_size=64
- Git commit: sandbox-ból nem lehet pusholni (nincs auth) → Mac terminálból kell: `git push`
- **±0.0 std determinizmus (KRITIKUS, MEGOLDVA v11-ben)**: Determinisztikus reset → policy mindig ugyanazt csinálja, eval szórás=0. Fix: reset noise_scale=0.01 (Humanoid-v4 mintájára). Ez volt a v8/v9/v10 plató gyökér oka.
- **86 lépéses fizikai határ (AKTÍV)**: A policy ~86 lépésnél konzisztensen elesik — ez nem reward döntés (break-even 10.8 lépés lenne), hanem fizikai instabilitás. A 2 sub-step fizikán tanult mozgásminta 86. lépésnél eléri stabilitási határát. Fix: sub-step 2→1 + fresh start (v13).
- **Catastrophic forgetting (MEGOLDVA v12b-ben)**: Reward komponens súly csökkentése (w_forward 8→4) finetune során → VecNormalize stat eltolódás → policy összeomlás (+133→-121). Fix: additív reward shaping (nem cserélni, hanem hozzáadni).
- **Finetune vs fresh start**: Ha egy policy "beégett" mozgásmintán ragad, finetune nem tudja felülírni. Fresh start szükséges az architektúrális változásokhoz (pl. sub-step szám).
- **Sub-step**: 2→1 (v13-ban). Lassabb fizika → robot tovább stabil → más mozgásmintát tanul a policy.
- **v17 reward súlyok (AKTUÁLIS)**: w_forward=8.0 (lineáris tracking visszaállítva!), w_air_time=3.0, w_dist=2.0, w_proximity=2.0, w_healthy=0.05, w_ctrl=-0.001, w_contact=-0.0001, w_fall=-20.0, w_stuck=-20.0, w_gait=0.0
- **v17 curriculum (10M)**: buoyancy 103N→0 (3M-7M), stuck_window 9999→40 (3M-7M)
- **v17 curriculum (20M)**: buoyancy 103N→0 (6M-14M), stuck_window 9999→40 (6M-14M)
- **v17 felhajtóerő**: 103N (konzervatív, 30%) — xfrc_applied[torso_id, 5], CurriculumCallback nullázza
- **v17 URDF figyelmeztetés**: G1-ben lehet virtuális torzó link; env betöltéskor kiírja a teljes tömeget — ellenőrizd!
- **v16 reward súlyok (archivált)**: w_vel=3.0 (Gaussian), w_air_time=3.0, stuck_window=40 — PPO ablak méretét tanulta
- **v15 reward súlyok (archivált)**: w_vel=3.0, w_air_time=1.0, w_stuck=-15.0, stuck_window=75
- **v12b reward súlyok (archivált)**: w_forward=8.0, w_dist=8.0 (potential-based, ÚJ), w_proximity=2.0 (3.5m küszöb, ÚJ), w_healthy=0.05, w_ctrl=-0.001, w_contact=-0.0001, w_fall=-20.0, w_gait=0.0
- **Tracking reward (v11+)**: `w_forward × dot(lin_vel[:2], direction_to_target)` — sebesség × célirány. Jobb mint y_velocity, mert minden irányban jutalmazza a haladást.
- **Env reward súlyok (jelenlegi v12)**:
  - `w_forward=4.0` (tracking: vel × dir, csökkentve 8→4)
  - `w_dist=5.0` (potential-based: prev_dist - curr_dist, ÚJ)
  - `w_proximity=3.0` (lineáris bonus 2.0m-en belül, ÚJ)
  - `w_healthy=0.05, w_ctrl=-0.001, w_contact=-0.0001, w_fall=-20.0, w_gait=0.0`
- **v11 reward súlyok**: w_forward=8.0, w_healthy=0.05, w_ctrl=-0.001, w_contact=-0.0001, w_fall=-20.0, w_gait=0.0

---

## GitHub

Repo: https://github.com/vorilevi/roboshelf-ai
Utolsó commit: "Fix: G1 Kaggle útvonal, find timeout javítás, mujoco_menagerie klón"
