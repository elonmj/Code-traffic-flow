²import numpy as np
import os

filepath = r"d:\Projets\Alibi\Code project\results\riemann_choc_simple_motos.npz"
if os.path.exists(filepath):
    data = np.load(filepath, allow_pickle=True)
    print(f"Keys in {filepath}: {list(data.keys())}")
else:
    print(f"File not found: {filepath}")
