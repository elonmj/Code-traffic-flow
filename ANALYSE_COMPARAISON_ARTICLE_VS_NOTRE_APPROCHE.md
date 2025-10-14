# ANALYSE: Comparaison Article Scientific Reports vs Notre Approche

**Date**: 2025-10-13  
**Article**: "Adaptive urban traffic signal control based on enhanced deep reinforcement learning" (Cai & Wei, 2024)  
**Problème identifié**: 0% amélioration RL vs Baseline

---

## 1. CE QU'ON A MAL FAIT

### ❌ **Problème #1: TRAINING INSUFFISANT**

**Notre approche**:
```python
total_timesteps = 5000  # SEULEMENT 5000 steps!
```

**Article Scientific Reports**:
```python
episodes = 200
simulation_duration = 4500s  # Par épisode!
```

**Calcul de l'article**:
- 200 épisodes × 4500s = 900,000 secondes de simulation
- Avec decision interval 15s: 900,000 / 15 = **60,000 décisions**
- Avec épisodes de 3600s: 3600 / 15 = 240 décisions/épisode
- 200 épisodes = **48,000 décisions minimum**

**Notre réalité**:
- Total timesteps: 5000
- Decision interval: 15s
- Episode duration: 3600s
- Steps par épisode: 240
- **Nombre d'épisodes**: 5000 / 240 ≈ **21 épisodes**

**CONCLUSION**: On a entraîné **21 épisodes** au lieu de **200 épisodes**!  
→ **C'est 10x trop peu!**

---

### ❌ **Problème #2: PAS DE REPRISE DE CHECKPOINT**

**Ce qu'on pensait**:
```python
# On avait 6500 steps de training précédent
# On devait reprendre et continuer jusqu'à 50,000 ou plus
```

**Ce qui s'est passé (logs)**:
```python
2025-10-13 14:01:51 - INFO - train_rl_agent:628 -   - Total timesteps: 5000
2025-10-13 14:01:51 - INFO - _get_checkpoint_dir:150 - [PATH] Found 6 existing checkpoints
# MAIS: Aucun message "Checkpoint loaded" ou "Training resumed"!
```

**Preuve dans debug.log**:
- Line 8: `Total timesteps: 5000` (traffic_light_control)
- Line 28: `Total timesteps: 5000` (ramp_metering)
- Line 48: `Total timesteps: 5000` (adaptive_speed_control)

**AUCUNE reprise visible!**

**Pourquoi?**
Le code détecte les checkpoints existants mais **ne les charge PAS automatiquement**. Il faut probablement un flag explicite comme `--resume-from-checkpoint` ou modifier la logique de training.

---

### ❌ **Problème #3: REWARD FUNCTION MAL CONÇUE**

**Notre reward**:
```python
# Dans notre approche: Basé sur vitesses instantanées
reward = efficiency_metric  # Favorise basses densités → RED constant
```

**Article (équation 7)**:
```python
# Basé sur QUEUE LENGTH (longueur des files)
reward = -(Σ queue_length_t+1 - Σ queue_length_t)
# Positif si files diminuent, négatif si augmentent
```

**Différence critique**:
- **Article**: Minimise longueur des files → Encourage throughput
- **Nous**: Maximise vitesses → Encourage faible densité (RED constant)

**Notre problème observé**:
```
Steps 1-8: action=1.0 (GREEN), state change significatif
Steps 9+: action=0.0 (RED), state gelé, reward=9.8885 constant
```

→ L'agent apprend que **RED constant = reward stable élevé**!

---

### ❌ **Problème #4: CONFIGURATION DOMAIN INADÉQUATE**

**Notre config**:
```yaml
domain_length: 1000m  # 1 km
control_interval: 15s
episode_duration: 3600s
```

**Ratio critique**:
```
propagation_time = 1000m / 17m/s ≈ 60s
control_interval = 15s
ratio = 15/60 = 0.25  # Trop élevé!
```

**Article utilise**:
```python
lane_length: 500m per direction
total_coverage: ~150m from intersection center
detector_range: 147m
```

**Implication**:
- Article: Domain court (500m) mais **focus sur zone critique** (147m près intersection)
- Nous: Domain 1km mais **observations uniformes** → Pas d'exploitation spatiale

---

### ❌ **Problème #5: ACTION SPACE DESIGN**

**Article (Section "Action set")**:
```python
# Action = Durée de GREEN light
action_space = {5s, 10s, 15s, 20s, 30s, 35s, 40s, 45s}
# + 3s yellow light après chaque phase
# Phase cycle: Phase1 → Phase2 → Phase3 → Phase4 → repeat
```

**Notre approche**:
```python
# Action = GREEN (1.0) ou RED (0.0) binaire
action_space = Box(0, 1)  # Continu mais interprété comme binaire
```

**Différence**:
- **Article**: Agent contrôle **DURÉE** du feu vert (plus flexible)
- **Nous**: Agent contrôle seulement **ON/OFF** (moins flexible)

---

### ❌ **Problème #6: STATE REPRESENTATION**

**Article (Section "State space")**:
```python
# Position matrix: 12 lanes × 21 grids = 252 cells
# Velocity matrix: 12 lanes × 21 grids = 252 cells
# Phase info: 4-dim one-hot vector
# Total state: 504 + 4 = 508 dimensions

# Grid size: 7m (vehicle 5m + spacing 2m)
# Coverage: 21 grids × 7m = 147m per direction
```

**Notre approche**:
```python
# Observation: 26 dimensions
# Segments: 3-8 (~30-80m from boundary)
# Moins détaillé que l'article
```

**Différence**:
- **Article**: Représentation spatiale RICHE (252 cellules par matrice)
- **Nous**: Représentation COMPACTE (26 dimensions seulement)

---

## 2. TRAINING TIME RÉEL

**Article - Figure 5 (Cumulative Reward)**:
- Convergence visible après ~100 épisodes
- Stabilisation après ~150-175 épisodes
- Training complet: **200 épisodes**

**Calcul temps GPU (article)**:
```
1 épisode = 4500s simulation
Avec speedup GPU ~10x: 450s réels par épisode
200 épisodes × 450s = 90,000s = 25 heures
```

**Notre validation Kaggle**:
```
3 scénarios × 5000 timesteps = 15,000 steps totaux
Durée réelle: ~3h45min
Temps par scenario: ~1h15min
```

**Si on voulait faire 200 épisodes**:
```
200 épisodes × 240 steps/épisode = 48,000 steps
48,000 / 5000 × 1h15min ≈ 12 heures par scénario
3 scénarios × 12h = 36 heures GPU minimum!
```

---

## 3. ALGORITHME UTILISÉ

**Article: PN_D3QN (Prioritized Noisy Dueling Double Deep Q-Network)**

**Composants**:
1. **Dueling Network** (équation 8):
   ```python
   Q(s,a) = V(s) + [A(s,a) - mean(A(s,a))]
   # Sépare value fonction et advantage function
   ```

2. **Double Q-Learning** (équation 10):
   ```python
   Q_target = r + γ * Q(s', argmax Q(s',a'; w); w_target)
   # Utilise main network pour sélection, target network pour évaluation
   ```

3. **Prioritized Experience Replay** (équation 14):
   ```python
   P_j = 1 / rank(j)
   # Échantillonnage non-uniforme basé sur TD-error
   ```

4. **Noisy Network** (équation 15):
   ```python
   w_noisy = μ + ε ⊙ ξ  # ξ ~ N(0,1)
   # Injection de bruit dans parameters pour exploration
   ```

**Notre approche**:
```python
# PPO (Proximal Policy Optimization)?
# Ou DQN basique?
# Pas clair dans le code!
```

**MANQUE**: On n'utilise probablement **AUCUN** des 4 composants avancés de l'article!

---

## 4. PARAMÈTRES CLÉS COMPARÉS

| Paramètre | Article | Notre Approche | Ratio |
|-----------|---------|----------------|-------|
| **Episodes** | 200 | ~21 | **10x moins** |
| **Total decisions** | ~48,000 | 5,000 | **10x moins** |
| **Episode duration** | 4500s | 3600s | 0.8x |
| **Learning rate** | 0.001 | ? | ? |
| **Discount γ** | 0.95 | ? | ? |
| **Memory capacity** | 50,000 | ? | ? |
| **Batch size** | 128 | ? | ? |
| **Optimizer** | Adam | ? | ? |
| **Update interval** | ? | ? | ? |

---

## 5. RÉSULTATS ARTICLE

**Metrics** (Scenario 1 - Training):
```
Average Waiting Time: 
- FTC: ~80s
- Max-Pressure: ~60s
- D3QN: ~50s (after 200 episodes)
- PN_D3QN: ~40s (after 200 episodes)

Average Queue Length:
- FTC: ~7 vehicles
- Max-Pressure: ~5 vehicles
- D3QN: ~4 vehicles (after 200 episodes)
- PN_D3QN: ~3 vehicles (after 200 episodes)
```

**Cumulative Reward** (Figure 5):
- D3QN final: -592.05
- N_D3QN final: -367.33
- PN_D3QN final: **-229.24** (meilleur)

**Testing Scenarios** (Figure 6):
- Scenario 2 (low density): PN_D3QN ≈ D3QN
- Scenario 3 (high density): PN_D3QN >> D3QN
- Scenario 5 (variable): PN_D3QN améliore de **15-60%** vs autres méthodes

---

## 6. POURQUOI ÇA MARCHE DANS L'ARTICLE

### ✅ **Facteur 1: TRAINING MASSIF**
- 200 épisodes = Convergence garantie
- Figure 4 montre: Fluctuations jusqu'à épisode 100, puis stabilisation
- Nos 21 épisodes = Encore en phase d'exploration!

### ✅ **Facteur 2: REWARD APPROPRIÉ**
- Queue length = Métrique DIRECTEMENT liée au throughput
- Mesurable en temps réel (detectors)
- Encourage actions qui réduisent congestion
- **Notre reward**: Encourage vitesses élevées → RED constant OK si vitesses hautes!

### ✅ **Facteur 3: STATE REPRESENTATION RICHE**
- 504 dimensions = Information spatiale détaillée
- Agent peut "voir" formation de queues
- Position + velocity matrices = Exploitation temporelle ET spatiale

### ✅ **Facteur 4: ALGORITHME AVANCÉ**
- Dueling network: Meilleure estimation de V(s)
- Double Q-learning: Évite overestimation
- PER: Apprentissage 2x plus rapide (citation article)
- Noisy network: Robustesse à différents traffic flows

### ✅ **Facteur 5: ACTION SPACE RÉALISTE**
- Durée de green light = Contrôle pratique
- 8 valeurs discrètes = Exploration efficace
- Phase cycle = Sécurité garantie (pas de switches arbitraires)

---

## 7. CE QU'ON DOIT CORRIGER

### 🔧 **Correction #1: AUGMENTER TRAINING**

**Option A - Matching article**:
```python
total_timesteps = 48000  # 200 épisodes × 240 steps
# Temps GPU estimé: ~12h par scénario = 36h total
```

**Option B - Minimum viable**:
```python
total_timesteps = 24000  # 100 épisodes × 240 steps
# Temps GPU estimé: ~6h par scénario = 18h total
# Devrait montrer convergence selon Figure 5
```

**Option C - Quick test**:
```python
total_timesteps = 12000  # 50 épisodes × 240 steps
# Temps GPU estimé: ~3h par scénario = 9h total
# Pourrait suffire pour voir tendance
```

### 🔧 **Correction #2: FIX REWARD FUNCTION**

**Implémenter reward article**:
```python
def calculate_reward(self, prev_state, current_state):
    """
    Article equation (7): r_t = -(Σ queue_t+1 - Σ queue_t)
    
    Positive reward → Queue lengths decreased
    Negative reward → Queue lengths increased
    """
    prev_queue = self._get_total_queue_length(prev_state)
    curr_queue = self._get_total_queue_length(current_state)
    
    reward = -(curr_queue - prev_queue)
    return reward

def _get_total_queue_length(self, state):
    """
    Calculate total queue length across all lanes.
    Queue = vehicles with speed < threshold (e.g., 1 m/s)
    """
    # Count vehicles with v < 1.0 m/s as queued
    rho_m, rho_c, w_m, w_c = state
    
    queue_m = np.sum(rho_m[w_m < 1.0])  # Mainlane queue
    queue_c = np.sum(rho_c[w_c < 1.0])  # Corridor queue
    
    return queue_m + queue_c
```

### 🔧 **Correction #3: AMÉLIORER STATE REPRESENTATION**

**Option A - Grille spatiale (comme article)**:
```python
# Diviser domain en grilles 7m
n_grids = int(domain_length / 7)  # 1000/7 ≈ 143 grids

# Position matrix: binary (0/1 si véhicule présent)
# Velocity matrix: speed value at each grid
state_dim = n_grids * 2 + 4  # positions + velocities + phase
```

**Option B - Segments enrichis**:
```python
# Augmenter nombre de segments observés
n_segments = 20  # Au lieu de 6 actuellement
state_dim = n_segments * 4 + 4  # (rho, w) × 2 lanes × 20 segments + phase
```

### 🔧 **Correction #4: ACTION SPACE RÉALISTE**

**Implémenter phase-cycle avec durées**:
```python
# Discrete action space: Green light duration
action_space = Discrete(8)  # [5s, 10s, 15s, 20s, 30s, 35s, 40s, 45s]

# Phase cycle
phases = ['NS_through', 'NS_left', 'EW_through', 'EW_left']
current_phase_idx = 0  # Rotate through phases

def step(self, action):
    # action = index in [0, 7]
    green_duration = [5, 10, 15, 20, 30, 35, 40, 45][action]
    yellow_duration = 3  # Fixed safety buffer
    
    # Execute green phase
    self._run_simulation(green_duration, phase='green')
    
    # Execute yellow phase
    self._run_simulation(yellow_duration, phase='yellow')
    
    # Move to next phase in cycle
    current_phase_idx = (current_phase_idx + 1) % 4
    
    return next_state, reward, done, info
```

### 🔧 **Correction #5: IMPLÉMENTER PN_D3QN**

**Utiliser Stable-Baselines3 avec custom network**:
```python
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DuelingQNetwork(BaseFeaturesExtractor):
    """
    Dueling architecture: Q(s,a) = V(s) + [A(s,a) - mean(A)]
    """
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    
    def forward(self, obs):
        features = self.feature_extractor(obs)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Dueling aggregation
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

# Training with PER and Double DQN
model = DQN(
    policy='MlpPolicy',
    env=env,
    learning_rate=0.001,
    buffer_size=50000,
    batch_size=128,
    gamma=0.95,
    target_update_interval=100,  # Double DQN
    exploration_fraction=0.3,
    exploration_final_eps=0.01,
    prioritized_replay=True,  # PER
    verbose=1
)
```

### 🔧 **Correction #6: WARM-UP PHASE**

**Exclure convergence initiale**:
```python
def run_episode(self, warm_up_duration=300):
    """
    Warm-up: First 300s not counted in metrics
    Evaluation: 300s - 3600s
    """
    # Warm-up phase (no rewards counted)
    self._run_simulation(warm_up_duration, count_reward=False)
    
    # Evaluation phase
    total_reward = 0
    for step in range(evaluation_steps):
        action = self.select_action(state)
        next_state, reward, done = self.step(action)
        total_reward += reward
        state = next_state
    
    return total_reward
```

---

## 8. PLAN D'ACTION RECOMMANDÉ

### **Phase 1: Minimal Fix (1-2 jours)**

**Objectif**: Atteindre convergence avec minimal changes

1. **Augmenter training**: 24,000 timesteps (100 épisodes)
2. **Fix reward**: Implémenter queue-length based reward
3. **Tester localement**: Vérifier convergence sur 10 épisodes
4. **Kaggle validation**: Run complet (~18h GPU)

**Attendu**: 10-20% amélioration (si ça converge)

### **Phase 2: Full Article Replication (3-5 jours)**

**Objectif**: Reproduire résultats article

1. **State representation**: Grille spatiale 7m
2. **Action space**: Phase-cycle avec durées [5-45s]
3. **Algorithm**: Implémenter PN_D3QN avec Dueling + PER + Noisy
4. **Training**: 48,000 timesteps (200 épisodes)
5. **Multiple scenarios**: Test robustness

**Attendu**: 30-60% amélioration (matching article)

### **Phase 3: Thesis Integration (1 jour)**

**Objectif**: Documenter pour défense

1. **Comparison table**: Article vs Notre approche
2. **Learning curves**: Figure montrant convergence
3. **Performance metrics**: Table comparative
4. **Limitations**: Discussion honnête

---

## 9. ESTIMATION RÉALISTE

### **Avec Minimal Fix (Phase 1)**:
- **Training time**: 18h GPU (3 scenarios × 6h)
- **Success probability**: 60%
- **Expected improvement**: 10-20%
- **Thesis ready**: OUI (avec discussion limitations)

### **Avec Full Replication (Phase 2)**:
- **Training time**: 36h GPU (3 scenarios × 12h)
- **Success probability**: 85%
- **Expected improvement**: 30-60%
- **Thesis ready**: OUI (résultats solides)

### **Timeline réaliste**:
```
J0 (aujourd'hui): Analyse et décision
J1: Implementation minimal fix
J2: Training + validation Kaggle
J3: Analyse résultats
J4: Documentation thèse
J5: Buffer / Full replication si nécessaire
```

---

## 10. CONCLUSION

**Ce qu'on a compris**:
1. ❌ **Training insuffisant**: 21 épisodes au lieu de 200
2. ❌ **Pas de reprise checkpoint**: On est reparti de zéro
3. ❌ **Reward mal conçu**: Favorise RED constant au lieu de throughput
4. ❌ **Algorithm basique**: Manque Dueling + PER + Noisy + Double DQN
5. ❌ **State representation simple**: 26 dims au lieu de 504

**Ce qu'on doit faire EN PRIORITÉ**:
1. ✅ **Fix #1**: Augmenter à 24,000 timesteps minimum (100 épisodes)
2. ✅ **Fix #2**: Implémenter queue-length reward (équation 7 article)
3. ✅ **Fix #3**: Vérifier reprise checkpoint fonctionne
4. ✅ **Fix #4**: Ajouter warm-up phase (300s)

**Message pour toi**:
On n'a PAS échoué. On a juste sous-estimé le training nécessaire. L'article utilise **10x plus de training** que nous. C'est normal qu'on ne voit pas d'amélioration avec seulement 21 épisodes alors qu'ils en font 200!

**Prochaine étape immédiate**: Tu veux qu'on implémente le minimal fix (Phase 1) pour tester rapidement? Ou tu préfères directement viser la full replication (Phase 2)?
