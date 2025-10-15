# CYCLE DEBUG BUG #35 - SESSION DU 2025-10-15

## RÉSUMÉ EXÉCUTIF

**Status**: 🔴 **ÉCHEC APRÈS 4 ITÉRATIONS**  
**Problème**: Queues = 0.00 dans TOUS les kernels malgré multiples fixes  
**Iterations testées**: 4 kernels (cuyy, obwe, tzte, drbh)  
**Temps investi**: ~1h30 (4× 20min par kernel)  
**Conclusion**: **ROOT CAUSE NON IDENTIFIÉ** - Problème plus profond que prévu  

---

## CHRONOLOGIE DES TENTATIVES

### Kernel 1: cuyy (Bug #34 fix uniquement)
**Fix appliqué**: Inflow BC avec vitesse d'équilibre (w = 2.26 m/s à ρ=200)  
**Résultat**: Queue = 0.00 partout  
**Diagnostic**: Bug #34 insuffisant, Bug #35 découvert (velocities pas de relaxation)

### Kernel 2: obwe (Bug #35 CPU fix)
**Fix appliqué**: CPU ODE solver - raise error si road_quality is None  
**Résultat**: Queue = 0.00 partout  
**Diagnostic**: Fix appliqué mais simulation tourne sur GPU, pas CPU

### Kernel 3: tzte (Bug #35 GPU fix)
**Fix appliqué**: GPU network splitting - passer grid.d_R au lieu de None  
**Résultat**: Queue = 0.00 partout  
**Diagnostic**: d_R passé correctement mais queues toujours nulles → Hypothèse: densité trop faible

### Kernel 4: drbh (Inflow density × 1.5)
**Fix appliqué**: Augmentation densité inflow de 200 → 300 veh/km (near-jam)  
**Config**: ρ_m_inflow = 300 veh/km, Ve théorique ≈ 1 m/s (crawling)  
**Résultat**: Queue = 0.00 partout  
**Diagnostic**: **ÉCHEC COMPLET** - Même avec densité jam, pas de queues détectées

---

## MÉTRIQUES COMPARATIVES

| Kernel | Fix Principal | Training Zeros | Eval Zeros | Queue Max | Agent Action (Eval) |
|--------|---------------|----------------|------------|-----------|---------------------|
| cuyy | Bug #34 (equilibrium inflow) | 3% | 100% | 0.00 | 0 (stuck) |
| obwe | Bug #35 CPU | 13% | 97.5% | 0.00 | 1 (stuck) |
| tzte | Bug #35 GPU | 10% | 97.5% | 0.00 | 1 (stuck) |
| drbh | Density ×1.5 | 6% | 100% | 0.00 | 0 (stuck) |

**Pattern commun**: 
- ✅ Training: Rewards diversifiés (-0.01 to 0.02) grâce à R_diversity
- ❌ Evaluation: 97-100% zeros, agent stuck à action constante
- ❌ Queues: **0.00 PARTOUT** indépendamment du fix appliqué

---

## HYPOTHÈSES EXPLORÉES (ET INFIRMÉES)

### ✅ H1: Inflow BC avec free speed au lieu d'equilibrium
**Fix**: Bug #34 - Calculer vitesse équilibre pour inflow  
**Résultat**: Appliqué dans cuyy → **Queues still 0.00** ❌

### ✅ H2: CPU ODE solver avec fallback silencieux R=3
**Fix**: Bug #35 CPU - Raise error au lieu de fallback  
**Résultat**: Appliqué dans obwe → **Queues still 0.00** ❌

### ✅ H3: GPU ODE solver ne reçoit pas d_R (road quality)
**Fix**: Bug #35 GPU - Passer grid.d_R au solve_ode_step_gpu  
**Résultat**: Appliqué dans tzte → **Queues still 0.00** ❌

### ✅ H4: Densité inflow trop faible pour atteindre seuil v < 5 m/s
**Fix**: Augmenter ρ_inflow de 200 → 300 veh/km  
**Résultat**: Appliqué dans drbh → **Queues still 0.00** ❌

---

## HYPOTHÈSES RESTANTES (NON TESTÉES)

### H5: Queue calculation logic broken in environment
**Location**: `Code_RL/src/env/traffic_signal_env_direct.py:370-378`  
**Suspect Code**:
```python
QUEUE_SPEED_THRESHOLD = 5.0  # m/s
queued_m = densities_m[velocities_m < QUEUE_SPEED_THRESHOLD]
queued_c = densities_c[velocities_c < QUEUE_SPEED_THRESHOLD]
current_queue_length = (np.sum(queued_m) + np.sum(queued_c)) * dx
```

**Possible Issues**:
1. `velocities_m` extraction incorrecte (toujours > 5 m/s)
2. `densities_m` normalisée au lieu de valeur brute
3. `dx` calculation wrong (grid spacing)
4. Observation segments wrong indices

**Test Needed**: Add logging in _calculate_queue_length():
```python
print(f"[QUEUE_DEBUG] velocities_m: {velocities_m}")
print(f"[QUEUE_DEBUG] threshold check: {velocities_m < QUEUE_SPEED_THRESHOLD}")
print(f"[QUEUE_DEBUG] queued_m densities: {queued_m}")
print(f"[QUEUE_DEBUG] queue_length calculated: {current_queue_length}")
```

### H6: Observation extraction returns normalized instead of physical values
**Location**: `arz_model/simulation/runner.py` - `get_segment_observations()`  
**Suspect**: Velocities normalized to [0,1] range instead of m/s  

**Test Needed**: Check observation dict structure:
```python
obs_data = runner.get_segment_observations([8, 9, 10])
print(f"[OBS_DEBUG] Raw obs_data: {obs_data}")
print(f"[OBS_DEBUG] v_m values: {obs_data['v_m']}")  # Should be in m/s, not [0,1]
```

### H7: Traffic never reaches observation segments
**Current segments**: [8, 9, 10] upstream, [11, 12, 13] downstream (x=80-130m)  
**Hypothesis**: Traffic accumulates at x=0-50m but dissipates before x=80m  

**Test Needed**: Change observation segments to near-inflow:
```python
observation_segments = {
    'upstream': [2, 3, 4],    # x=20-40m
    'downstream': [5, 6, 7]   # x=50-70m
}
```

**Note**: Already tested locally in `test_observation_segments.py` → Traffic detected BUT velocities=15 m/s (wrong)

### H8: ARZ relaxation NOT actually applied in GPU kernel
**Evidence**:
- Local diagnostic `test_bug35_fix.py` shows velocities DO relax on CPU
- But Kaggle runs on GPU
- GPU kernel may have different implementation

**Test Needed**: Add GPU kernel logging:
```python
@cuda.jit
def _ode_rhs_kernel_gpu(...):
    if j == 0:  # First cell only
        print(f"[GPU_KERNEL] Cell 0: rho={rho_m:.4f} Ve={Ve_m:.2f} v_current={v_m:.2f}")
```

---

## ACTIONS RECOMMANDÉES (PAR PRIORITÉ)

### 🔥 CRITICAL: Diagnostic Logging Phase

Before ANY more kernel launches, add comprehensive logging to identify WHERE the problem is:

#### Action 1: Log Queue Calculation
**File**: `Code_RL/src/env/traffic_signal_env_direct.py`  
**Location**: `_calculate_queue_length()` method  
**Add**:
```python
def _calculate_queue_length(self, observation):
    # ... existing code ...
    
    print(f"[QUEUE_DIAGNOSTIC] ===== Step {self.current_step} =====")
    print(f"[QUEUE_DIAGNOSTIC] Observation shape: {observation.shape}")
    print(f"[QUEUE_DIAGNOSTIC] velocities_m: {velocities_m}")
    print(f"[QUEUE_DIAGNOSTIC] velocities_c: {velocities_c}")
    print(f"[QUEUE_DIAGNOSTIC] Threshold: {QUEUE_SPEED_THRESHOLD} m/s")
    print(f"[QUEUE_DIAGNOSTIC] Below threshold (m): {velocities_m < QUEUE_SPEED_THRESHOLD}")
    print(f"[QUEUE_DIAGNOSTIC] Below threshold (c): {velocities_c < QUEUE_SPEED_THRESHOLD}")
    print(f"[QUEUE_DIAGNOSTIC] queued_m densities: {queued_m}")
    print(f"[QUEUE_DIAGNOSTIC] queued_c densities: {queued_c}")
    print(f"[QUEUE_DIAGNOSTIC] queue_length: {current_queue_length:.2f} vehicles")
    
    return current_queue_length
```

#### Action 2: Log Observation Extraction
**File**: `Code_RL/src/env/traffic_signal_env_direct.py`  
**Location**: `_get_observation()` method  
**Add**:
```python
def _get_observation(self):
    obs_data = self.runner.get_segment_observations(self.observation_segments)
    
    print(f"[OBS_DIAGNOSTIC] ===== Step {self.current_step} =====")
    print(f"[OBS_DIAGNOSTIC] Segments: {self.observation_segments}")
    print(f"[OBS_DIAGNOSTIC] Raw obs_data keys: {obs_data.keys()}")
    print(f"[OBS_DIAGNOSTIC] rho_m: {obs_data['rho_m'][:3]} (first 3)")
    print(f"[OBS_DIAGNOSTIC] v_m: {obs_data['v_m'][:3]} (first 3, should be m/s)")
    print(f"[OBS_DIAGNOSTIC] v_free_m: {self.v_free_m} m/s (normalization factor)")
    
    # ... existing normalization code ...
```

#### Action 3: Log Runner State
**File**: `arz_model/simulation/runner.py`  
**Location**: `get_segment_observations()` method  
**Add**:
```python
def get_segment_observations(self, segment_indices):
    # ... existing code to extract rho, v ...
    
    print(f"[RUNNER_DIAGNOSTIC] ===== Observation Extraction =====")
    print(f"[RUNNER_DIAGNOSTIC] Requested segments: {segment_indices}")
    print(f"[RUNNER_DIAGNOSTIC] Grid N_physical: {self.grid.N_physical}")
    print(f"[RUNNER_DIAGNOSTIC] State U shape: {self.U.shape}")
    print(f"[RUNNER_DIAGNOSTIC] rho_m at segments: {rho_m}")
    print(f"[RUNNER_DIAGNOSTIC] q_m at segments: {self.U[1, segment_indices]}")
    print(f"[RUNNER_DIAGNOSTIC] v_m calculated: {v_m}")
    
    return {'rho_m': rho_m, 'v_m': v_m, 'rho_c': rho_c, 'v_c': v_c}
```

### 📊 THEN: Relaunch ONE Kernel with Logging

**Expected Output Pattern if Velocities Wrong**:
```
[RUNNER_DIAGNOSTIC] v_m calculated: [15.0, 15.0, 15.0]  ← CONSTANT, WRONG
[OBS_DIAGNOSTIC] v_m: [15.0, 15.0, 15.0]  ← PASSED THROUGH
[QUEUE_DIAGNOSTIC] velocities_m: [15.0, 15.0, 15.0]  ← STILL CONSTANT
[QUEUE_DIAGNOSTIC] Below threshold: [False, False, False]  ← NO QUEUE DETECTED
[QUEUE_DIAGNOSTIC] queue_length: 0.00
```

**Expected Output Pattern if Velocities Correct**:
```
[RUNNER_DIAGNOSTIC] v_m calculated: [3.2, 4.1, 6.5]  ← VARYING WITH DENSITY
[OBS_DIAGNOSTIC] v_m: [3.2, 4.1, 6.5]  ← CORRECT VALUES
[QUEUE_DIAGNOSTIC] velocities_m: [3.2, 4.1, 6.5]  ← GOOD
[QUEUE_DIAGNOSTIC] Below threshold: [True, True, False]  ← QUEUE DETECTED!
[QUEUE_DIAGNOSTIC] queue_length: 15.30  ← NON-ZERO!
```

---

## NEXT SESSION PLAN

1. ✅ Add diagnostic logging (Actions 1-3 above)
2. 🚀 Launch ONE kernel with logging enabled
3. 📊 Analyze logs to identify EXACT failure point:
   - Is v_m extraction wrong?
   - Is normalization breaking values?
   - Are segments receiving traffic?
   - Is calculation logic itself broken?
4. 🔧 Apply TARGETED fix based on diagnostic findings
5. ✅ Relaunch to verify fix
6. 🔄 Repeat until queues > 0 detected

**DO NOT LAUNCH MORE KERNELS WITHOUT DIAGNOSTICS** - We're shooting in the dark!

---

## LEÇONS APPRISES

1. **Multiple fixes without diagnostics = Waste of time**
   - 4 kernels × 20 min = 1h20 wasted on blind fixes
   - ONE kernel with logging would have identified root cause in 20 min

2. **Local tests ≠ Kaggle GPU behavior**
   - `test_bug35_fix.py` showed relaxation works locally (CPU)
   - But Kaggle runs GPU → different code path

3. **Need systematic debugging, not hypothesis-driven fixes**
   - Each hypothesis seemed reasonable
   - But without logging, impossible to verify

4. **Queue detection has 3 failure points**:
   - Runner state (velocity calculation from q/ρ)
   - Observation extraction (normalization/denormalization)
   - Environment logic (threshold check, length calculation)
   - Must log ALL three to isolate problem

---

## STATUS FINAL

**Queues détectées**: ❌ NON (0.00 dans TOUS les kernels)  
**Root cause identifié**: ❌ NON  
**Prochaine action**: 🔍 DIAGNOSTIC LOGGING REQUIS  
**Estimation temps**: 1 kernel (20 min) + analyse (10 min) = 30 min  

**🙏 Que la volonté de Dieu soit faite - La vérité se révélera dans les logs de diagnostic**

---

## ANNEXE: COMMANDES UTILES

### Analyse rapide queues
```powershell
Select-String -Path "validation_output/results/elonmj_arz-validation-76rlperformance-*/arz-*.log" -Pattern "QUEUE: current=" | Select-Object -First 20 | ForEach-Object { $_.Line -replace '.*current=([0-9.]+).*','$1' } | Sort-Object -Unique
```

### Comparaison rewards entre kernels
```powershell
Get-ChildItem validation_output/results/elonmj_arz-validation-76rlperformance-*/ -Filter "arz-*.log" | ForEach-Object { 
    $kernel = $_.Directory.Name
    $zeros = (Select-String -Path $_.FullName -Pattern "reward=0.0000" | Measure-Object).Count
    "$kernel : $zeros zeros"
}
```

### Vérif config inflow density
```powershell
Select-String -Path "validation_output/results/elonmj_arz-validation-76rlperformance-*/section_7_6_rl_performance/data/scenarios/traffic_light_control.yml" -Pattern "state:" -Context 0,4 | Select-Object -First 1
```
