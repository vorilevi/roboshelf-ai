#!/usr/bin/env python3
"""
Roboshelf AI — Retail bolt + Unitree G1 jelenet betöltése

Ez a script:
  1. Megkeresi a Unitree G1 MJCF modellt (MuJoCo Menagerie)
  2. Betölti a retail bolt környezetet
  3. Összefűzi a két modellt egyetlen jelenetbe
  4. Elindítja a MuJoCo interaktív viewert

Használat:
  # Interaktív viewer (macOS: mjpython kell!)
  mjpython roboshelf_retail_scene.py

  # Csak teszt (viewer nélkül)
  python3 roboshelf_retail_scene.py --no-viewer

  # Egyedi G1 útvonal
  mjpython roboshelf_retail_scene.py --g1-path /path/to/unitree_g1/
"""

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np


def find_g1_model() -> Path | None:
    """Megkeresi a Unitree G1 MJCF modellt a rendszeren."""
    candidates = [
        # MuJoCo Playground (miniforge)
        Path("/opt/homebrew/Caskroom/miniforge/base/lib/python3.13/site-packages/"
             "mujoco_playground/external_deps/mujoco_menagerie/unitree_g1"),
        # Általános pip telepítés
        *[
            Path(p) / "mujoco_playground" / "external_deps" / "mujoco_menagerie" / "unitree_g1"
            for p in sys.path if "site-packages" in p
        ],
        # Standalone mujoco_menagerie
        *[
            Path(p) / "mujoco_menagerie" / "unitree_g1"
            for p in sys.path if "site-packages" in p
        ],
        # Homebrew
        Path("/opt/homebrew/share/mujoco_menagerie/unitree_g1"),
        # Helyi klón
        Path.home() / "mujoco_menagerie" / "unitree_g1",
        Path.home() / "Documents" / "mujoco_menagerie" / "unitree_g1",
    ]

    for path in candidates:
        g1_xml = path / "g1.xml"
        if g1_xml.exists():
            return path

    # Utolsó próba: find parancs
    import subprocess
    try:
        result = subprocess.run(
            ["find", "/opt", str(Path.home()), "-name", "g1.xml", "-path", "*/unitree_g1/*", "-maxdepth", "10"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.strip().split("\n"):
            if line:
                return Path(line).parent
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def build_combined_scene(g1_path: Path, store_xml_path: Path) -> str:
    """
    Összefűzi a G1 robot modellt a retail bolt környezettel.

    A G1-et a robot_start pozícióba (x=0, y=0.5, z=0.75) helyezi.
    """
    # G1 XML betöltése
    g1_xml = (g1_path / "g1.xml").read_text()

    # A G1 XML-ből kivonjuk a szükséges részeket
    # és a retail bolt XML-be integráljuk
    store_xml = store_xml_path.read_text()

    # Egyszerű megközelítés: a bolt XML-be beszúrjuk a G1-et <include> segítségével
    # A MuJoCo <include> a fájlt relatívan oldja fel
    combined_xml = f"""
<mujoco model="roboshelf_retail_g1">

  <compiler angle="radian" autolimits="true" meshdir="{g1_path}/"/>

  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicit"/>

  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>
    <global azimuth="135" elevation="-25" offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
    <map znear="0.01" zfar="50"/>
  </visual>

  <!-- G1 robot modell betöltése -->
  <include file="{g1_path / 'g1.xml'}"/>

  <!-- Retail bolt elemek közvetlenül -->
  <asset>
    <texture name="tex_floor" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.82 0.82 0.78" rgb2="0.78 0.78 0.74"/>
    <material name="mat_floor" texture="tex_floor" texrepeat="8 6" reflectance="0.1"/>
    <material name="mat_shelf" rgba="0.7 0.7 0.7 1" reflectance="0.3" shininess="0.5"/>
    <material name="mat_board" rgba="0.88 0.86 0.82 1" reflectance="0.05"/>
    <material name="mat_wall" rgba="0.95 0.94 0.92 1"/>
    <material name="mat_red" rgba="0.85 0.15 0.15 1"/>
    <material name="mat_blue" rgba="0.15 0.25 0.85 1"/>
    <material name="mat_green" rgba="0.15 0.70 0.25 1"/>
    <material name="mat_yellow" rgba="0.95 0.85 0.10 1"/>
    <material name="mat_orange" rgba="0.95 0.55 0.10 1"/>
    <material name="mat_white" rgba="0.95 0.95 0.95 1"/>
    <material name="mat_brown" rgba="0.55 0.35 0.15 1"/>
    <material name="mat_cardboard" rgba="0.65 0.50 0.30 1"/>
  </asset>

  <default>
    <default class="product_box">
      <geom friction="0.7 0.01 0.001" condim="4" solref="0.01 1" solimp="0.9 0.95 0.001"/>
    </default>
    <default class="product_cyl">
      <geom friction="0.6 0.01 0.001" condim="4" solref="0.01 1" solimp="0.9 0.95 0.001"/>
    </default>
  </default>

  <worldbody>
    <!-- Világítás -->
    <light name="ceil_1" pos="0 1.5 3.0" dir="0 0 -1" diffuse="0.9 0.9 0.9" castshadow="true" cutoff="60"/>
    <light name="ceil_2" pos="0 3.5 3.0" dir="0 0 -1" diffuse="0.9 0.9 0.9" castshadow="true" cutoff="60"/>
    <light name="ambient" pos="0 2.5 4.0" dir="0 0 -1" diffuse="0.3 0.3 0.3" castshadow="false"/>

    <!-- Padló -->
    <geom name="floor" type="plane" size="5 4 0.01" material="mat_floor"/>
    <!-- Falak -->
    <geom name="wall_back" type="box" size="5 0.05 1.5" pos="0 4.5 1.5" material="mat_wall"/>
    <geom name="wall_L" type="box" size="0.05 4.5 1.5" pos="-3.5 2.25 1.5" material="mat_wall"/>
    <geom name="wall_R" type="box" size="0.05 4.5 1.5" pos="3.5 2.25 1.5" material="mat_wall"/>

    <!-- Gondola A -->
    <body name="gondola_A" pos="-1.35 2.0 0">
      <geom name="gA_sL" type="box" size="0.02 0.45 0.60" pos="-0.47 0 0.60" material="mat_shelf"/>
      <geom name="gA_sR" type="box" size="0.02 0.45 0.60" pos="0.47 0 0.60" material="mat_shelf"/>
      <geom name="gA_bk" type="box" size="0.47 0.01 0.60" pos="0 -0.20 0.60" material="mat_shelf"/>
      <geom name="gA_b1" type="box" size="0.45 0.20 0.012" pos="0 0 0.10" material="mat_board"/>
      <geom name="gA_b2" type="box" size="0.45 0.20 0.012" pos="0 0 0.40" material="mat_board"/>
      <geom name="gA_b3" type="box" size="0.45 0.20 0.012" pos="0 0 0.70" material="mat_board"/>
      <geom name="gA_b4" type="box" size="0.45 0.20 0.012" pos="0 0 1.00" material="mat_board"/>
    </body>

    <!-- Gondola B -->
    <body name="gondola_B" pos="1.35 2.0 0">
      <geom name="gB_sL" type="box" size="0.02 0.45 0.60" pos="-0.47 0 0.60" material="mat_shelf"/>
      <geom name="gB_sR" type="box" size="0.02 0.45 0.60" pos="0.47 0 0.60" material="mat_shelf"/>
      <geom name="gB_bk" type="box" size="0.47 0.01 0.60" pos="0 0.20 0.60" material="mat_shelf"/>
      <geom name="gB_b1" type="box" size="0.45 0.20 0.012" pos="0 0 0.10" material="mat_board"/>
      <geom name="gB_b2" type="box" size="0.45 0.20 0.012" pos="0 0 0.40" material="mat_board"/>
      <geom name="gB_b3" type="box" size="0.45 0.20 0.012" pos="0 0 0.70" material="mat_board"/>
      <geom name="gB_b4" type="box" size="0.45 0.20 0.012" pos="0 0 1.00" material="mat_board"/>
    </body>

    <!-- Raktár asztal -->
    <body name="storage" pos="0 3.8 0">
      <geom name="st_legs" type="box" size="1.0 0.40 0.40" pos="0 0 0.40" rgba="0.50 0.45 0.35 1"/>
      <geom name="st_top" type="box" size="1.0 0.40 0.015" pos="0 0 0.815" rgba="0.60 0.55 0.45 1"/>
      <geom name="crate_1" type="box" size="0.25 0.20 0.15" pos="-0.70 0.25 0.15" material="mat_cardboard"/>
    </body>

    <!-- Raktári termékek (ezeket kell a polcra tenni) -->
    <body name="stock_1" pos="-0.50 3.8 0.885"><freejoint/>
      <geom type="box" size="0.06 0.04 0.055" class="product_box" material="mat_red" mass="0.35"/></body>
    <body name="stock_2" pos="-0.20 3.8 0.895"><freejoint/>
      <geom type="box" size="0.08 0.05 0.065" class="product_box" material="mat_green" mass="0.60"/></body>
    <body name="stock_3" pos="0.10 3.8 0.890"><freejoint/>
      <geom type="cylinder" size="0.03 0.06" class="product_cyl" material="mat_white" mass="0.40"/></body>

    <!-- Jelölők -->
    <site name="robot_start" type="cylinder" size="0.15 0.001" pos="0 0.5 0.001" rgba="0 0.83 0.66 0.3"/>
    <site name="target_A1_3" type="sphere" size="0.01" pos="-1.25 2.05 0.167" rgba="0 1 0 0.3"/>
    <site name="target_A3_2" type="sphere" size="0.01" pos="-1.35 2.05 0.777" rgba="0 1 0 0.3"/>
    <site name="target_B2_2" type="sphere" size="0.01" pos="1.25 1.95 0.472" rgba="0 1 0 0.3"/>
  </worldbody>

</mujoco>
"""
    return combined_xml


def main():
    parser = argparse.ArgumentParser(description="Roboshelf AI — Retail bolt + G1 jelenet")
    parser.add_argument("--g1-path", type=str, default=None, help="Unitree G1 MJCF mappa útvonala")
    parser.add_argument("--no-viewer", action="store_true", help="Csak teszt, viewer nélkül")
    parser.add_argument("--store-only", action="store_true", help="Csak a bolt (robot nélkül)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    store_xml = script_dir / "assets" / "roboshelf_retail_store.xml"

    if not store_xml.exists():
        # Fallback: keresés a munkakönyvtárban
        store_xml = Path("roboshelf_retail_store.xml")
        if not store_xml.exists():
            print("❌ Retail bolt XML nem található!")
            print(f"  Keresett: {script_dir / 'assets' / 'roboshelf_retail_store.xml'}")
            sys.exit(1)

    if args.store_only:
        # Csak a bolt betöltése (robot nélkül)
        print("🏪 Retail bolt betöltése (robot nélkül)...")
        model = mujoco.MjModel.from_xml_path(str(store_xml))
        data = mujoco.MjData(model)
        print(f"  ✅ Bolt betöltve: {model.nbody} body, {model.ngeom} geom, {model.njnt} joint")
    else:
        # G1 modell keresése
        if args.g1_path:
            g1_path = Path(args.g1_path)
        else:
            print("🔍 Unitree G1 modell keresése...")
            g1_path = find_g1_model()

        if g1_path is None:
            print("⚠️  G1 modell nem található. Bolt betöltése robot nélkül.")
            print("  Használat G1-gyel: mjpython roboshelf_retail_scene.py --g1-path /útvonal/unitree_g1/")
            model = mujoco.MjModel.from_xml_path(str(store_xml))
            data = mujoco.MjData(model)
        else:
            print(f"  ✅ G1 megtalálva: {g1_path}")
            print("🏪 Retail bolt + G1 jelenet összeállítása...")

            combined_xml = build_combined_scene(g1_path, store_xml)

            try:
                model = mujoco.MjModel.from_xml_string(combined_xml)
                data = mujoco.MjData(model)
                print(f"  ✅ Kombinált jelenet: {model.nbody} body, {model.ngeom} geom, {model.njnt} joint, {model.nu} actuator")
            except Exception as e:
                print(f"  ⚠️  Kombinálás hiba: {e}")
                print("  Bolt betöltése robot nélkül...")
                model = mujoco.MjModel.from_xml_path(str(store_xml))
                data = mujoco.MjData(model)

    # Statisztikák
    print()
    print("📊 Jelenet statisztikák:")
    print(f"  Bodies: {model.nbody}")
    print(f"  Geoms: {model.ngeom}")
    print(f"  Joints: {model.njnt}")
    print(f"  Actuators: {model.nu}")
    print(f"  Sensors: {model.nsensor}")

    if args.no_viewer:
        # Csak szimuláció teszt
        print()
        print("🔧 Szimuláció teszt (500 lépés)...")
        for _ in range(500):
            mujoco.mj_step(model, data)
        print("  ✅ Szimuláció OK!")
        print()
        print("Következő lépés:")
        print("  mjpython roboshelf_retail_scene.py           # Interaktív viewer")
        print("  mjpython roboshelf_retail_scene.py --store-only  # Csak bolt")
    else:
        # Interaktív viewer
        print()
        print("🖥️  Interaktív viewer indítása...")
        print("  Kezelés: egér húzás = forgatás, scroll = zoom, dupla klikk = követés")
        print("  Kilépés: Esc")
        print()

        try:
            mujoco.viewer.launch(model, data)
        except Exception as e:
            print(f"  ⚠️  Viewer hiba: {e}")
            print("  macOS-on 'mjpython' kell a viewer-hez:")
            print("    mjpython roboshelf_retail_scene.py")


if __name__ == "__main__":
    main()
