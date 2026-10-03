"""Copy the mechanisms that are too large for the repository from Cantera's data.

    python tools/cantera_mechanisms.py

Writes mechanisms/external/n-hexane-NUIG-2015.yaml (Zhang et al. 2015, 1268
species), the large mechanism of the chemistry benchmarks and of V3.
"""
import os
import shutil

import cantera as ct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["example_data/n-hexane-NUIG-2015.yaml"]


def main():
    out = os.path.join(ROOT, "mechanisms", "external")
    os.makedirs(out, exist_ok=True)
    for name in FILES:
        for directory in ct.get_data_directories():
            path = os.path.join(directory, name)
            if os.path.exists(path):
                shutil.copy(path, os.path.join(out, os.path.basename(name)))
                print(f"copied {path}")
                break
        else:
            raise SystemExit(f"{name} not found in Cantera {ct.__version__}'s data directories")


if __name__ == "__main__":
    main()
