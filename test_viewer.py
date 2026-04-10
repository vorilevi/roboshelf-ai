import mujoco
import mujoco.viewer
from pathlib import Path

model_path = Path("src/envs/assets/roboshelf_retail_store.xml")
print("Model betöltése:", model_path)

m = mujoco.MjModel.from_xml_path(str(model_path))
d = mujoco.MjData(m)

print("Model betöltve, bodies:", m.nbody, "geoms:", m.ngeom)

with mujoco.viewer.launch_passive(m, d) as v:
    print("Viewer elindult, fut a főciklus...")
    while v.is_running():
        mujoco.mj_step(m, d)
        v.sync()
