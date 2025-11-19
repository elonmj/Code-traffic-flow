# Optimisations Kaggle - Sans Logging (Respect à St Thomas d'Aquin)

**Date**: 2025-11-19  
**Branche**: `experiment/kaggle-rl-training`  
**Philosophie**: Investigation par lecture de code, pas instrumentation

---

## ✅ Optimisations Implémentées

### 1. ⏱️ Timeout Configurable

**Problème**: Timeout hardcodé → impossible pour le user de contrôler  
**Solution**: Ajout de `timeout_seconds` dans les configs Pydantic

#### Code Modifié

**`arz_model/config/time_config.py`**:
```python
timeout_seconds: Optional[float] = Field(
    default=None,
    description="Maximum wall-clock time in seconds (None = infinite). Useful for Kaggle kernels."
)
```

**`Code_RL/training/config/training_config.py`**:
```python
class EvaluationStrategy(BaseModel):
    timeout_seconds: Optional[float] = Field(
        default=None, 
        description="Timeout max (s) pour évaluation (None = infini)"
    )

def kaggle_gpu_config(...):
    return TrainingConfig(
        evaluation_strategy=EvaluationStrategy(
            timeout_seconds=300.0  # 5 min pour éviter inactivité Kaggle
        )
    )
```

**Impact**: User peut maintenant définir `timeout_seconds=300.0` pour éviter timeouts Kaggle

---

### 2. 🚀 Réutilisation d'Env dans Sanity Checker

**Problème Identifié** (analyse Kaggle log):
- Check #4 (rollout): Crée env → 1 reconstruction (12MB GPU)
- Check #5 (reward diversity): Crée NOUVEL env → 1 reconstruction (2MB GPU)
- **Total gaspillage**: 2 reconstructions, ~5-7 secondes, 30% du temps setup

**Solution Architecturale** (sans logging):
- Créer **UN SEUL** env partagé
- Passer l'env à checks #4 et #5
- Fermer après tous les checks

#### Code Modifié

**`Code_RL/training/core/sanity_checker.py`**:
```python
def run_all_checks(self) -> List[SanityCheckResult]:
    # ...checks 1-3...
    
    # 🚀 OPTIMIZATION: Réutiliser UN SEUL env pour checks #4 et #5
    shared_env = None
    if self.sanity_config.enabled:
        shared_env = self._create_test_env()  # Créer UNE FOIS
    
    if self.sanity_config.enabled:
        self.results.append(self._check_environment_rollout(env=shared_env))
    
    if self.sanity_config.enabled:
        self.results.append(self._check_reward_diversity(env=shared_env))
    
    # Cleanup
    if shared_env is not None:
        shared_env.close()
```

**Signatures Modifiées**:
```python
def _check_environment_rollout(self, env=None) -> SanityCheckResult:
    env_created = False
    if env is None:
        env = self._create_test_env()
        env_created = True
    # ... tests ...
    finally:
        if env_created:
            env.close()

def _check_reward_diversity(self, env=None) -> SanityCheckResult:
    # Même pattern
```

**Gains Estimés**:
- **Reconstructions**: 9 → 8 (-11%)
- **Temps setup**: 44s → 39s (-5s, -11%)
- **Overhead total**: 30% → 26% (-4 points)

---

## 🔍 Investigation make_vec_env() (Sans Logging)

### Méthodologie: Lecture de Code Source SB3

**Chemin**: `C:\Users\JOSAPHAT\AppData\Roaming\Python\Python312\site-packages\stable_baselines3\common\env_util.py`

**Code Critique**:
```python
def make_vec_env(env_id, n_envs=1, ...):
    vec_env = vec_env_cls([make_env(i) for i in range(n_envs)], **vec_env_kwargs)
    return vec_env
```

### Résultats d'Investigation

**Avec `n_envs=1`**:
- ✅ Crée **EXACTEMENT 1** environnement (list comprehension)
- ✅ Pas de vérifications cachées trouvées
- ✅ Pas de warm-up automatique

**EvalCallback**:
```python
# Code_RL/training/core/trainer.py:389
eval_callback = EvalCallback(
    self.eval_env,  # ← Reçoit env existant
    ...
)
```
- ✅ Reçoit `self.eval_env` déjà créé
- ✅ Ne crée PAS d'env supplémentaire

### Conclusion: Mystère des 6-7 Envs Inexpliqués

**Hypothèses** (sans pouvoir confirmer sans instrumentation):
1. **GPU Pool Warm-up**: Première allocation GPU déclenche plusieurs tentatives internes
2. **Python Garbage Collection**: Objets temporaires créés/détruits pendant import
3. **Numba JIT**: Compilation kernels GPU peut construire objets temporaires
4. **SB3 Internal Checks**: Possible vérification silencieuse non visible dans env_util.py

**Décision**: Ne PAS ajouter de logging (respect philosophie). Se concentrer sur causes **confirmées** (sanity_checker).

---

## 📊 Comparaison Avant/Après

### Kaggle Log Original (commit d67602f)
```
Timeline Breakdown:
- Setup overhead: 56% (83.3s) - sanity checks, imports, network builds
- Wasteful reconstructions: 29% (44s) - 7 unnecessary builds
- Useful training: 11% (16s) - actual learning
- Evaluation: 4% (5.7s) - model validation
```

**Reconstructions Détaillées**:
- Sanity check #4 (rollout): 1 reconstruction ❌
- Sanity check #5 (reward): 1 reconstruction ❌  
- make_vec_env() mystery: 6-7 reconstructions ❓ (non résolu)
- Training env: 1 reconstruction ✅ (nécessaire)
- Eval env: 1 reconstruction ✅ (nécessaire)

### Après Optimisation (estimé)
```
Timeline Breakdown:
- Setup overhead: 52% (78s) - sanity checks optimisées
- Wasteful reconstructions: 26% (39s) - 6 unnecessary builds (-1 build)
- Useful training: 11% (16s) - unchanged
- Evaluation: 4% (5.7s) - unchanged
- TOTAL: ~150s → ~145s (-5s, -3.3%)
```

**Reconstructions Optimisées**:
- Sanity checks: 1 reconstruction ✅ (partagé entre #4 et #5)
- make_vec_env() mystery: 6-7 reconstructions ❓ (non résolu)
- Training env: 1 reconstruction ✅
- Eval env: 1 reconstruction ✅

---

## 🏗️ Architectures Long Terme (TODO)

### 1. GPU Pool Singleton Pattern
**Objectif**: Cache global pour éviter réallocations GPU

```python
class GPUMemoryPoolSingleton:
    _instance = None
    _pool = None
    
    @classmethod
    def get_pool(cls, segment_ids, N_per_segment):
        if cls._pool is None:
            cls._pool = GPUMemoryPool(segment_ids, N_per_segment)
        return cls._pool
```

**Gain estimé**: -20-30s (éliminer 6-7 reconstructions mystérieuses)

### 2. Environment Factory Pooling
**Objectif**: Réutiliser envs au lieu de reconstruire

```python
class EnvironmentPool:
    def __init__(self, factory, max_size=3):
        self.factory = factory
        self.available = []
        self.in_use = []
    
    def acquire(self):
        if self.available:
            return self.available.pop()
        return self.factory()
    
    def release(self, env):
        env.reset()
        self.available.append(env)
```

**Gain estimé**: -10-15s (réutiliser envs entre sanity checks et training)

---

## 🎯 Prochaines Étapes

1. **Tester optimisations** sur Kaggle
2. **Mesurer gains réels** (sans logging, juste timing total)
3. **Si gains insuffisants**: Implémenter GPU Pool Singleton
4. **Documentation**: Mettre à jour RL_TRAINING_SURVIVAL_GUIDE.md

---

## 📝 Notes Philosophiques

**Respect à St Thomas d'Aquin**: "Entia non sunt multiplicanda praeter necessitatem"  
→ Ne pas multiplier les logs sans nécessité. Investigation par LECTURE, pas INSTRUMENTATION.

**Approche adoptée**:
- ✅ Lire code source SB3
- ✅ Analyser patterns d'allocation
- ✅ Optimiser causes confirmées
- ❌ PAS de print() debugging
- ❌ PAS de logging supplémentaire

Cette approche philosophique force une compréhension PROFONDE du code plutôt qu'une dépendance aux logs.
