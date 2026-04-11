# Roboshelf AI — Session Kontextus

> Ezt a fájlt az AI olvassa, hogy gyorsan felvegye a fonalat.
> Utoljára frissítve: 2026-04-10 (fresh start Mac tanítás indítva)

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

**MacBook M2 CPU** — `m2_10m_v8` szint, 10M lépés, ACTION_SCALE=0.3
- Script: `src/training/roboshelf_phase2_train.py --level m2_10m_v8`
- Indítás: `cd ~/roboshelf-ai-dev/roboshelf-ai && python src/training/roboshelf_phase2_train.py --level m2_10m_v8`
- Becsült idő: ~1 óra (M2 CPU, 4 env, ~2900 FPS)
- **Fő fix: ACTION_SCALE=0.3 — max ±0.3 radian eltérés az egyensúlytól**
- Reward: w_forward=5.0, w_healthy=1.0, w_fall=-20.0, w_gait=0.0

**Kaggle T4** — leállítva (n_envs=8 hiba + GPU kihasználtság korlátai)

---

## Következő teendők (prioritás sorrendben)

1. **Mac fresh start eredmény kiértékelése** — 3M lépés után
   - Ha reward >+50 és ep hossz >50: jó alap, finetune-olható
   - Ha plató < -50 marad: w_gait csökkenteni, w_forward növelni
2. **GitHub push** — commitelni kell Mac terminálból: `git add -A && git commit -m "Fresh start m2_3m_fresh szint" && git push`
3. **Fázis 3 tanítóscript megírása** — `src/envs/roboshelf_manipulation_env.py` már megvan,
   csak a `src/training/roboshelf_phase3_train.py` hiányzik

---

## Fontos fájlok

```
src/envs/roboshelf_retail_nav_env.py       ← Fázis 2 env (survival bónusz w=0.5 hozzáadva)
src/envs/roboshelf_manipulation_env.py     ← Fázis 3 env (vázlat, kész)
src/envs/assets/roboshelf_retail_store.xml ← Bolt MJCF (2 gondola, termékek)
src/training/roboshelf_phase2_train.py     ← Fázis 2 tanítás (m2_3m_fresh szint hozzáadva)
src/training/roboshelf_phase2_finetune.py  ← Fine-tune script (evaluations.npz append-del, flush fix)
notebooks/roboshelf_phase2_kaggle.ipynb    ← Kaggle notebook (KaggleFlushCb + SyncNormCb fix)
src/roboshelf_phase2_check.py              ← Rendszerellenőrző
roboshelf-results/phase2/models/best/      ← Legjobb modell (7.8M lépés, reward=+42, ep=50)
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

**Áttörés:** 7.8M lépésnél a reward pozitívba fordult (+42) és az ep hossz áttörte a 43 lépéses plafont (50 lépés).
**Contact pattern bevezetése (12.2M+):** visszaesés -84-re, majd plató 52 ep hossznál. A gait reward (w=0.18) túl gyenge volt a régi modell "szokásaival" szemben → fresh start szükséges.
**Jelenlegi legjobb korábbi Mac modell:** 10.2M lépés, reward=+51, ep hossz=51 (`roboshelf-results/phase2/models/best/best_model.zip`)
**Fresh start (m2_3m_fresh, w_gait=0.18):** 2026-04-10-én indítva. 3M+6M lépés, de 43 ep hossznál megragadt. A gait reward konfliktusban volt a korai járásminta kialakításával.
**Tanulság:** Gait reward-ot csak akkor szabad bevezetni, ha a robot már tud járni (reward > 0, ep hossz > 100). Curriculum megközelítés szükséges.
**Következő fresh start (m2_3m_nogait, w_gait=0.0):** Gait kikapcsolva. Ha 3M után reward > -50, finetune-olni és csak akkor bevezetni a gait reward-ot.

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

---

## GitHub

Repo: https://github.com/vorilevi/roboshelf-ai
Utolsó commit: "Fix: G1 Kaggle útvonal, find timeout javítás, mujoco_menagerie klón"
