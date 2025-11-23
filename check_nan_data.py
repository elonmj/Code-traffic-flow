import numpy as np
from pathlib import Path

base_dir = Path("d:/Projets/Alibi/Code project/results/thesis_stage1/")

files = [
    'riemann_choc_simple_motos.npz',
    'riemann_detente_voitures.npz',
    'riemann_apparition_vide_motos.npz',
    'riemann_discontinuite_contact.npz',
    'riemann_interaction_multiclasse.npz'
]

print("Checking for NaN values in Riemann data files:")
print("=" * 70)

for fname in files:
    fpath = base_dir / fname
    if not fpath.exists():
        print(f"{fname}: FILE NOT FOUND")
        continue
        
    data = np.load(fpath, allow_pickle=True)
    
    rho_c = data['rho_c_history']
    rho_m = data['rho_m_history']
    v_c = data['v_c_history']
    v_m = data['v_m_history']
    
    has_nan_rho_c = np.isnan(rho_c).any()
    has_nan_rho_m = np.isnan(rho_m).any()
    has_nan_v_c = np.isnan(v_c).any()
    has_nan_v_m = np.isnan(v_m).any()
    
    print(f"\n{fname}:")
    print(f"  rho_c: shape={rho_c.shape}, has_NaN={has_nan_rho_c}, min={np.nanmin(rho_c):.6e}, max={np.nanmax(rho_c):.6e}")
    print(f"  rho_m: shape={rho_m.shape}, has_NaN={has_nan_rho_m}, min={np.nanmin(rho_m):.6e}, max={np.nanmax(rho_m):.6e}")
    print(f"  v_c: shape={v_c.shape}, has_NaN={has_nan_v_c}, min={np.nanmin(v_c):.6e}, max={np.nanmax(v_c):.6e}")
    print(f"  v_m: shape={v_m.shape}, has_NaN={has_nan_v_m}, min={np.nanmin(v_m):.6e}, max={np.nanmax(v_m):.6e}")
    
    if has_nan_rho_c or has_nan_rho_m or has_nan_v_c or has_nan_v_m:
        # Find when NaN first appears
        for t_idx in range(rho_c.shape[0]):
            if np.isnan(rho_c[t_idx, :]).any() or np.isnan(v_c[t_idx, :]).any():
                print(f"  🚨 First NaN at timestep {t_idx}/{rho_c.shape[0]} (t={data['t_history'][t_idx]:.2f}s)")
                break

print("\n" + "=" * 70)
