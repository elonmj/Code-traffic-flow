# 🚀 KAGGLE QUICK START - EXPERIMENT A

## 📋 Création du Notebook

**Titre**: `ARZ Validation - Experiment A - No Behavioral Coupling - v11`

**Settings**:
- Accélérateur: **GPU P100** 
- Internet: **ON**
- Persistence: Files only

---

## 💻 Code à copier/coller

### Cell 1: Clone & Setup
```python
# Clone repository
!git clone https://github.com/elonmj/Code-traffic-flow.git
%cd Code-traffic-flow

# Checkout experiment branch
!git checkout experiment/no-behavioral-coupling
!git log --oneline -5

# Verify modification
!grep -A 5 "EXPERIMENT A" arz_model/network/network_grid.py | head -10
```

### Cell 2: Install Dependencies
```python
# Install required packages
!pip install numba matplotlib -q

# Verify Numba CUDA
import numba.cuda as cuda
print(f"CUDA available: {cuda.is_available()}")
if cuda.is_available():
    print(f"GPU: {cuda.get_current_device().name.decode()}")
```

### Cell 3: Run Test
```python
# Change to scripts directory
%cd validation_ch7/scripts

# Run full test (15s simulation)
!python test_gpu_stability.py

# Note: Attendu ~8 minutes d'exécution
# 150,000 timesteps @ ~0.003s/step
```

### Cell 4: Check Results
```python
# Display figure
from IPython.display import Image, display
display(Image('/kaggle/working/results/gpu_stability_evolution.png'))

# Show metrics
import json
with open('/kaggle/working/results/gpu_stability_metrics.json', 'r') as f:
    metrics = json.load(f)
    print(json.dumps(metrics, indent=2))

# Show verdict
with open('/kaggle/working/session_summary.json', 'r') as f:
    summary = json.load(f)
    print(f"\n{'='*80}")
    print(f"VERDICT: {'SUCCESS' if summary['success'] else 'FAILURE'}")
    print(f"{'='*80}")
```

---

## 📊 Résultats à surveiller

### Console Output - Clés à chercher:

```
🔬 EXPERIMENT A VERDICT: θ COUPLING HYPOTHESIS TEST
================================================================================

✅ SUCCESS: Simulation STABLE without θ coupling!
   → Lagrangian θ_k coupling IS THE ROOT CAUSE!

OU

❌ FAILURE: Instability PERSISTS even without θ coupling!
   → θ coupling is NOT the root cause
```

### Métriques critiques:

```json
{
  "final_v_max": ???,      // < 20 m/s = SUCCESS
  "final_rho_max": ???,    // > 0.08 = Congestion
  "stable": ???,           // true = SUCCESS
  "success": ???           // Overall verdict
}
```

---

## 🎯 Comparaison Kernel 10 vs 11

| Métrique | Kernel 10 (AVEC θ) | Kernel 11 (SANS θ) | Verdict |
|----------|-------------------|-------------------|---------|
| v_max | 172.11 m/s | ??? m/s | ??? |
| t_explosion | 1.0s (6.7%) | ??? | ??? |
| Stable? | ❌ Non | ??? | ??? |

---

## ⏱️ Timing estimé

```
Setup (clone, install): ~1 min
Simulation: ~7-8 min
Plotting/save: ~30 sec
Total: ~9 minutes
```

---

## 📥 Fichiers à télécharger

Après exécution:

```bash
# Depuis Kaggle Output
/kaggle/working/results/
├── gpu_stability_evolution.png
├── gpu_stability_metrics.json
└── session_summary.json
```

**Action**: Download all → envoyer au user pour analyse!

---

## ✅ Checklist Exécution

- [ ] Notebook créé avec titre correct
- [ ] GPU P100 sélectionné
- [ ] Cell 1: Clone OK
- [ ] Cell 2: Numba CUDA detected
- [ ] Cell 3: Simulation lancée
- [ ] Monitoring: Check console every 2 min
- [ ] Cell 4: Results displayed
- [ ] Files downloaded
- [ ] Verdict interprété

---

**READY TO LAUNCH!** 🚀
