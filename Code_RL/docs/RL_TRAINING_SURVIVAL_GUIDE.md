# 🚨 GUIDE DE SURVIE: Entraînement RL Traffic Signals

**Source**: Analyse de 351 commits, 34 bugs documentés (branche `experiment/no-behavioral-coupling`)  
**Date**: 18 Novembre 2025  
**Objectif**: **Ne JAMAIS répéter ces erreurs fatales**

---

## ⚡ Les 5 Bugs Qui Ruinent L'Entraînement RL

### 1. 🔴 BUG #37: `int(action)` Au Lieu de `round(action)`

**Impact**: CRITIQUE - Apprentissage impossible

**Problème**:
```python
# ❌ MAUVAIS
self.current_phase = int(action)  # Tronque TOUJOURS vers 0

# Actions continues: 0.3, 0.5, 0.7, 0.95 → TOUTES deviennent 0 (RED)
# → Agent bloqué en phase RED → Reward = 0 → Pas de gradient
```

**Solution**:
```python
# ✅ CORRECT
self.current_phase = round(float(action))  # Arrondit au seuil 0.5

# 0.3 → 0 (RED), 0.51 → 1 (GREEN), 0.7 → 1 (GREEN) ✓
```

**Vérification**:
```python
# Test unitaire à ajouter AVANT l'entraînement
def test_action_mapping():
    actions = [0.0, 0.3, 0.5, 0.7, 0.95, 1.0]
    expected = [0, 0, 0, 1, 1, 1]  # round() behavior
    for action, expected_phase in zip(actions, expected):
        assert round(float(action)) == expected_phase
```

---

### 2. 🔴 BUG #33: Flux Entrant < Flux Initial

**Impact**: CRITIQUE - Pas de trafic à gérer

**Problème**:
```python
# Configuration qui évacue le trafic PAR LA GAUCHE!
rho_initial = 125 veh/km × v=5.33 m/s = 0.666 veh/s  # Flux initial
rho_inflow = 200 veh/km × v=2.67 m/s = 0.534 veh/s   # Flux entrant

# 0.534 < 0.666 → Onde de raréfaction → Queue = 0.00 TOUJOURS ❌
```

**Solution**:
```python
# ✅ Route vide au début, forte demande à l'entrée
rho_initial = max_density * 0.1  # 10% léger (free-flow)
w_initial = free_speed_m  # Vitesse libre

rho_inflow = max_density * 0.8  # 80% demande
w_inflow = free_speed_m  # Arrivant à vitesse

# Flux: q_inflow >> q_initial → Queue se forme naturellement ✓
```

**Vérification**:
```python
# Logs microscopiques - queue DOIT croître
# Step 1: queue=0.00
# Step 5: queue=2.50  ← DOIT augmenter!
# Step 10: queue=12.30
```

---

### 3. 🔴 BUG #27: Intervalle Contrôle = Temps Propagation

**Impact**: CRITIQUE - 0% amélioration

**Problème**:
```python
domain_length = 1000m
wave_speed = 17 m/s
propagation_time = 1000 / 17 ≈ 59s

control_interval = 60s  # ❌ Ratio ≈ 1.0 → Régime stationnaire!
```

**Conséquence**: Système atteint état stationnaire AVANT chaque décision → Contrôle inefficace

**Solution**:
```python
# ✅ Intervalle beaucoup plus court
control_interval = 15s  # Littérature: 5-15s optimal

# Ratio: 15s / 59s = 0.25 ← Agent peut exploiter la dynamique transitoire
```

---

### 4. 🔴 BUG Reward: Reward Function = 0.0 Toujours

**Impact**: CRITIQUE - Pas de signal d'apprentissage

**Symptômes**:
```python
# Logs montrent TOUJOURS:
# Reward: 0.00, 0.00, 0.00, 0.00, ...
# Unique values: 1 (devrait être > 10)
```

**Causes Possibles**:
1. Queue toujours zéro (BUG #33)
2. Delta queue = 0 (régime stationnaire, BUG #27)
3. Action mapping cassé (BUG #37)
4. Logique reward inversée/mal implémentée

**Solution**:
```python
# Logs microscopiques OBLIGATOIRES
def step(self, action):
    # ... state update ...
    
    # ✅ TOUJOURS logger les composants de reward
    print(f"Queue: prev={prev_queue:.2f}, current={current_queue:.2f}, delta={delta_queue:.4f}")
    print(f"Reward components: R_queue={r_queue:.4f}, R_switches={r_switches:.4f}")
    print(f"TOTAL REWARD: {reward:.4f}")
    
    return obs, reward, done, info
```

---

### 5. 🔴 BUG 0%: Fenêtres Temporelles Différentes

**Impact**: CRITIQUE - Comparaison invalide

**Problème**:
```python
# ❌ Baseline et RL évaluent sur des périodes DIFFÉRENTES
baseline_duration = 600s  # 10 min
rl_duration = 3600s       # 1 heure

# Même si RL fonctionne, métriques sont incomparables!
```

**Solution**:
```python
# ✅ MÊME configuration pour baseline et RL
EVAL_CONFIG = {
    "duration": 3600.0,  # IDENTIQUE
    "control_interval": 15.0,  # IDENTIQUE
    "seed": 42,  # IDENTIQUE pour reproductibilité
}

baseline_result = run_baseline(**EVAL_CONFIG)
rl_result = run_rl_agent(**EVAL_CONFIG)
```

---

## ✅ Checklist Pré-Entraînement (OBLIGATOIRE)

Avant de lancer `train_dqn.py`, vérifier:

- [ ] **Actions**: `round(float(action))` utilisé (PAS `int()`)
- [ ] **Flux**: `q_inflow >> q_initial` vérifié mathématiquement
- [ ] **Intervalle**: `control_interval = 15s` (PAS 60s)
- [ ] **Reward logs**: Imprime queue, delta, composants reward à chaque step
- [ ] **Test rapide**: 100 steps avec actions aléatoires → rewards DOIVENT varier
- [ ] **Fenêtres identiques**: Baseline et RL même duration/interval/seed

---

## 🧪 Test de Sanité Pré-Entraînement

```python
# À exécuter AVANT l'entraînement réel
def sanity_check(env, num_steps=100):
    """Vérifie que l'environnement peut générer des rewards variés"""
    
    rewards = []
    queues = []
    
    env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()  # Actions aléatoires
        obs, reward, done, info = env.step(action)
        
        rewards.append(reward)
        queues.append(info.get('queue_length', 0))
        
        if done:
            env.reset()
    
    # CHECKS OBLIGATOIRES
    unique_rewards = len(set(rewards))
    max_queue = max(queues)
    
    assert unique_rewards > 5, f"❌ Rewards trop uniformes! Unique values: {unique_rewards}"
    assert max_queue > 5.0, f"❌ Queue jamais formée! Max: {max_queue}"
    assert not all(r == 0 for r in rewards), "❌ Tous les rewards = 0!"
    
    print(f"✅ Sanity check PASSED:")
    print(f"   - Unique rewards: {unique_rewards}")
    print(f"   - Queue range: {min(queues):.2f} → {max(queues):.2f}")
    print(f"   - Reward range: {min(rewards):.4f} → {max(rewards):.4f}")
```

---

## 📊 Métriques de Succès

**Pendant l'entraînement** (logs à surveiller):

```
✅ BON SIGNE:
  - Reward varie: min=-2.5, max=1.2, mean=0.15
  - Queue varie: 0.0 → 50.0 véhicules
  - Phases changent: RED (15 steps) ↔ GREEN (20 steps)

❌ MAUVAIS SIGNE:
  - Reward = 0.00 constant (vérifier BUG #37, #33, #27)
  - Queue = 0.00 constant (vérifier BUG #33)
  - Phase = 0 constant (vérifier BUG #37)
```

---

## 🎓 Leçons Architecturales

### Modularité (de `niveau4_rl_performance/`)

```
training/
├── core/           # Business logic (agnostic au framework)
│   ├── controllers.py      # Baseline vs RL
│   ├── evaluators.py       # Métriques
│   └── cache_manager.py    # Cache intelligent
├── infrastructure/ # Implémentations techniques
│   ├── rl/                 # Stable-baselines3
│   ├── checkpoint/         # Rotation, validation
│   └── logging/            # Microscopique + TensorBoard
└── entry_points/   # CLI/Scripts
    └── train.py            # Point d'entrée unique
```

### Décisions Mathématiques (de `test_section_7_6_rl_performance.py`)

**Métriques de performance**:
```python
# Efficiency = sortie / entrée (devrait être < 1 en congestion)
efficiency = total_outflow / total_inflow

# Delay = temps passé - temps free-flow
delay = travel_time - free_flow_time

# Improvement = (baseline - rl) / baseline * 100
improvement_pct = (baseline_metric - rl_metric) / baseline_metric * 100
```

---

## 🚀 Workflow Recommandé

1. **Sanity check** (5 min): Actions aléatoires → rewards doivent varier
2. **Quick test** (15 min, 5000 steps): Vérifier apprentissage début
3. **Production run** (2-4h, 100k steps): Entraînement complet
4. **Évaluation**: Comparer vs baseline sur MÊME fenêtre temporelle

---

**Dernier conseil**: Si reward = 0.0 constant après 1000 steps → ARRÊTER, debugger, ne pas perdre de temps !
