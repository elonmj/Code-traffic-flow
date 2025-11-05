# Rapport de Recherche: Implémentation des Kernels GPU pour ARZ Model

**Date:** 2025-11-04  
**Objectif:** Documenter les recherches et découvertes avant implémentation des kernels GPU purs

---

## 🔍 DÉCOUVERTES CRITIQUES

### 1. **CODE GPU EXISTANT AVEC NUMBA CUDA** 
✅ **Le projet utilise déjà Numba CUDA (pas CuPy)** pour les kernels GPU!

**Fichiers GPU existants:**
```
arz_model/numerics/gpu/
├── ssp_rk3_cuda.py          # Intégrateur SSP-RK3 avec kernels Numba
├── weno_cuda.py              # Reconstruction WENO5 GPU
└── utils.py                  # Utilitaires GPU
```

**Fonctions GPU déjà implémentées:**
1. `solve_ode_step_gpu()` - Ligne 715 dans `time_integration.py`
   - Utilise Numba CUDA kernels (`@cuda.jit`)
   - Opère sur `cuda.devicearray.DeviceNDArray`
   - **Déjà fonctionnel!**

2. `solve_hyperbolic_step_ssprk3_gpu()` - Ligne 1519
   - Utilise la classe `SSP_RK3_GPU` 
   - Support WENO5 + Godunov via kernels existants
   - **Déjà fonctionnel!**

3. Classe `SSP_RK3_GPU` dans `ssp_rk3_cuda.py`
   - 3 kernels: `ssp_rk3_stage1_kernel`, `stage2_kernel`, `stage3_kernel`
   - Gère l'orchestration des 3 étapes SSP-RK3
   - **Complètement implémenté!**

### 2. **PROBLÈME IDENTIFIÉ: MÉLANGE NUMBA/CUPY**

Notre implémentation temporaire dans `strang_splitting_step_gpu()` (ligne 1244) fait:
```python
import cupy as cp
U_cpu = cp.asnumpy(U_gpu)  # ❌ Transfert GPU→CPU
U_star = solve_ode_step_cpu(U_cpu, dt/2, ...)  # ❌ Calcul CPU
U_new_gpu = cp.asarray(U_new)  # ❌ Transfert CPU→GPU
```

**Problème:** Mélange CuPy (NetworkGrid) avec Numba CUDA (kernels existants)
- CuPy arrays (`cp.ndarray`) ≠ Numba arrays (`cuda.devicearray.DeviceNDArray`)
- Transferts CPU↔GPU inutiles à chaque timestep

---

## 📊 ARCHITECTURE ACTUELLE vs CIBLE

### État Actuel (Hybride)
```
NetworkGrid (CuPy)
    ↓ cp.asnumpy()
CPU Memory
    ↓ solve_ode_step_cpu()
CPU Computation
    ↓ cp.asarray()
NetworkGrid (CuPy)
```
**Performance:** ~23 minutes pour test partiel (interrupted)

### Architecture Cible (Pure GPU)
```
NetworkGrid (CuPy)
    ↓ CuPy → Numba conversion
Numba GPU kernels
    ↓ solve_ode_step_gpu()
    ↓ solve_hyperbolic_step_ssprk3_gpu()
Numba computation (GPU)
    ↓ Numba → CuPy conversion
NetworkGrid (CuPy)
```
**Performance attendue:** 2-10x speedup (élimine transferts CPU)

---

## 🛠️ SOLUTIONS IDENTIFIÉES

### Solution 1: Conversion CuPy ↔ Numba (RECOMMANDÉE)

**Via CUDA Array Interface:**
```python
# CuPy → Numba (zero-copy)
cp_array = cp.ndarray(...)
numba_array = cuda.as_cuda_array(cp_array)

# Numba → CuPy (zero-copy)
numba_array = cuda.device_array(...)
cp_array = cp.asarray(numba_array)
```

**Source:** CuPy/Numba sont compatibles via [CUDA Array Interface](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html)

### Solution 2: Standardiser sur Numba uniquement

Remplacer NetworkGrid CuPy par Numba partout:
```python
# Au lieu de:
segment['U_gpu'] = cp.asarray(U)

# Utiliser:
segment['U_gpu'] = cuda.to_device(U)
```

**Avantage:** Cohérence totale  
**Inconvénient:** Refactoring plus important de NetworkGrid

---

## 📚 MEILLEURES PRATIQUES GPU (Recherche Web)

### 1. CuPy Documentation
**Source:** https://docs.cupy.dev/en/stable/user_guide/kernel.html

**3 types de kernels CuPy:**
- `ElementwiseKernel`: Opérations élément par élément (similaire à broadcasting NumPy)
- `ReductionKernel`: Opérations de réduction (sum, max, etc.)
- `RawKernel`: Kernels CUDA C/C++ bruts (maximum de contrôle)

**Exemple ElementwiseKernel:**
```python
squared_diff = cp.ElementwiseKernel(
    'float32 x, float32 y',  # Input
    'float32 z',              # Output
    'z = (x - y) * (x - y)',  # Operation
    'squared_diff'            # Name
)
```

### 2. Performance Best Practices
**Source:** https://docs.cupy.dev/en/stable/user_guide/performance.html

**Benchmarking:**
```python
from cupyx.profiler import benchmark
print(benchmark(my_func, (a,), n_repeat=20))
# Output: CPU: 44.407 us, GPU-0: 181.565 us
```

**Optimisations clés:**
1. **Minimiser transferts CPU↔GPU** (notre problème actuel!)
2. **Utiliser CUB backend** pour réductions: `CUPY_ACCELERATORS=cub`
3. **Batch operations** plutôt qu'opérations séquentielles
4. **Overlapping work** avec streams CUDA

### 3. Numba CUDA
**Source:** https://numba.readthedocs.io/en/stable/cuda/index.html

**Kernel signature:**
```python
@cuda.jit
def my_kernel(input_arr, output_arr, N):
    i = cuda.grid(1)  # Thread index
    if i < N:
        output_arr[i] = input_arr[i] * 2
```

**Device functions (réutilisables):**
```python
@cuda.jit(device=True)
def helper_function(x):
    return x * x

@cuda.jit
def kernel_using_helper(arr, N):
    i = cuda.grid(1)
    if i < N:
        arr[i] = helper_function(arr[i])
```

---

## 🎯 PLAN D'IMPLÉMENTATION RECOMMANDÉ

### Phase 1: Conversion CuPy ↔ Numba (PRIORITÉ 1)
**Fichier:** `arz_model/numerics/time_integration.py`

Modifier `strang_splitting_step_gpu()`:
```python
def strang_splitting_step_gpu(U_gpu_cupy, dt, grid, params, seg_id=None):
    """GPU Strang splitting using existing Numba kernels."""
    from numba import cuda
    import cupy as cp
    
    # Convert CuPy → Numba (zero-copy via CUDA Array Interface)
    U_gpu_numba = cuda.as_cuda_array(U_gpu_cupy)
    
    # Step 1: ODE (dt/2) - GPU via existing kernel
    d_R = cuda.to_device(grid.road_quality[grid.physical_cell_indices])
    U_star_numba = solve_ode_step_gpu(U_gpu_numba, dt/2, grid, params, d_R)
    
    # Step 2: Hyperbolic (dt) - GPU via existing kernel
    U_ss_numba = solve_hyperbolic_step_ssprk3_gpu(U_star_numba, dt, grid, params, None)
    
    # Step 3: ODE (dt/2) - GPU via existing kernel
    U_new_numba = solve_ode_step_gpu(U_ss_numba, dt/2, grid, params, d_R)
    
    # Convert Numba → CuPy (zero-copy)
    U_new_cupy = cp.asarray(U_new_numba)
    
    return U_new_cupy
```

**Temps estimé:** 30-60 minutes  
**Gain attendu:** 5-10x speedup (élimine transferts CPU)

### Phase 2: Tests et Validation (PRIORITÉ 2)
**Fichier:** `tests/test_gpu_small_timestep.py`

1. Vérifier que le test passe avec GPU purs
2. Benchmarker CPU vs GPU avec `cupyx.profiler.benchmark`
3. Valider stabilité (v_max < 20 m/s, rho > 0.08)

**Temps estimé:** 15-30 minutes

### Phase 3: Optimisations Avancées (OPTIONNEL)
Si besoin de plus de performance:
1. Profiler avec Nsight Systems
2. Optimiser les kernels existants (shared memory, coalescing)
3. Overlapping computation avec streams CUDA

---

## ✅ ACTIONS IMMÉDIATES

1. **Implémenter conversion CuPy↔Numba** dans `strang_splitting_step_gpu()`
   - Utiliser `cuda.as_cuda_array()` et `cp.asarray()`
   - Appeler `solve_ode_step_gpu()` et `solve_hyperbolic_step_ssprk3_gpu()` existants

2. **Tester avec `pytest tests/test_gpu_small_timestep.py`**
   - Vérifier que v_max reste stable (<20 m/s)
   - Mesurer le temps d'exécution

3. **Benchmarker performance**
   - Comparer avec version CPU
   - Documenter les gains

---

## 📖 RÉFÉRENCES

1. **CuPy User Guide - Kernels**  
   https://docs.cupy.dev/en/stable/user_guide/kernel.html

2. **CuPy Performance Best Practices**  
   https://docs.cupy.dev/en/stable/user_guide/performance.html

3. **Numba CUDA Documentation**  
   https://numba.readthedocs.io/en/stable/cuda/index.html

4. **CUDA Array Interface (Interop CuPy/Numba)**  
   https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html

5. **Existing GPU Code in Project:**
   - `arz_model/numerics/gpu/ssp_rk3_cuda.py`
   - `arz_model/numerics/gpu/weno_cuda.py`
   - `arz_model/numerics/time_integration.py` (lines 715, 1519)

---

## 💡 CONCLUSION

**DÉCOUVERTE MAJEURE:** Le code GPU existe déjà avec Numba CUDA! Notre problème n'est pas d'implémenter les kernels depuis zéro, mais de **connecter correctement** le frontend CuPy (NetworkGrid) avec les kernels Numba existants.

**Solution:** Utiliser CUDA Array Interface pour conversion zero-copy entre CuPy et Numba.

**Temps d'implémentation:** 1-2 heures maximum (pas plusieurs jours!)

**Gain de performance attendu:** 5-10x speedup en éliminant les transferts CPU↔GPU actuels.
