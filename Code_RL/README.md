# Système de Contrôle de Signalisation par Apprentissage par Renforcement
# Adaptation Victoria Island Lagos, Nigeria

## 📋 Vue d'ensemble

Ce projet implémente un système complet de contrôle de feux de signalisation basé sur l'apprentissage par renforcement (RL), spécialement adapté pour le corridor Victoria Island à Lagos, Nigeria. Le système utilise un algorithme Deep Q-Network (DQN) pour optimiser la gestion du trafic dans un environnement urbain dense avec un mix de véhicules (voitures et motos).

**Architecture Modernisée (2025)**: Le système utilise maintenant une architecture Pydantic + couplage direct GPU pour des performances 100-200x supérieures à l'architecture HTTP précédente.

## 🌍 Contexte - Victoria Island Lagos

Victoria Island est un quartier central d'affaires de Lagos caractérisé par :
- **Trafic dense multi-modal** : 35% de motos, 65% de voitures
- **Intersections complexes** : Akin Adesola Street x Adeola Odeku Street
- **Routes hiérarchisées** : Primary (3 voies, 50 km/h), Secondary (2 voies, 40 km/h)
- **Comportements de conduite ouest-africains** : Gap-filling, dépassements fréquents

## 🏗️ Architecture du Système

### Composants Principaux

```
┌─────────────────────────────────────────────────────────┐
│                    Agent DQN                           │
│                (Stable-Baselines3)                     │
└─────────────┬───────────────────────────────────────────┘
              │ Actions (0: maintenir, 1: changer)
              ▼
┌─────────────────────────────────────────────────────────┐
│         TrafficSignalEnvDirectV2                       │
│   (Pydantic Config + Direct GPU Coupling)              │
└─────────────┬───────────────────────────────────────────┘
              │ Direct In-Process Memory Access
              ▼
┌─────────────────────────────────────────────────────────┐
│           SimulationRunner (arz_model)                 │
│           NetworkGrid + GPU Arrays                      │
└─────────────────────────────────────────────────────────┘
```

**Performance**:
- Step latency: ~0.2-0.6ms (vs 50-100ms HTTP-based)
- Episode throughput: ~1000+ steps/sec (vs 10-20 steps/sec)
- Memory: Direct GPU array access (no serialization)

### 1. ARZ Simulator (Arz-Zuriguel Model) - arz_model
- **Modèle de trafic multi-classe** supportant motos et voitures
- **Couplage direct GPU** : accès mémoire in-process (pattern MuJoCo)
- **Pydantic configuration** : type-safe, validated config objects
- **NetworkGrid** : multi-segment network simulation

### 2. Environnement Gymnasium (TrafficSignalEnvDirectV2)
- **Configuration Pydantic** : NetworkSimulationConfig from factory
- **Espace d'observation** : [ρ_m, v_m, ρ_c, v_c] × N_segments + phase_onehot (normalized [0,1])
- **Espace d'actions** : 2 actions discrètes (maintenir/changer phase)
- **Fonction de récompense** : R = -α·congestion + μ·throughput - κ·phase_change
- **Performance** : ~0.2-0.6ms per step (100-200x faster than HTTP)

### 3. Agent DQN (Stable-Baselines3)
- **Réseau de neurones** : Architecture adaptée aux 43 observations
- **Exploration** : ε-greedy avec décroissance
- **Mémoire de replay** : Buffer d'expériences pour stabilité
- **Target network** : Mise à jour périodique pour stabilité

## 📂 Structure du Projet

```
Code_RL/
├── 📁 src/                    # Code source principal
│   ├── 📁 config/            # Configuration Pydantic (NEW)
│   │   ├── __init__.py       # Exports: create_rl_training_config
│   │   └── rl_network_config.py # Factory RL-specific
│   ├── 📁 env/              # Environnement RL Gymnasium
│   │   ├── traffic_signal_env_direct.py # Legacy (deprecated)
│   │   └── traffic_signal_env_direct_v2.py # Modern Pydantic version
│   ├── 📁 rl/               # Algorithmes apprentissage
│   │   ├── train_dqn.py     # Entraînement DQN
│   │   └── baseline.py      # Baselines (fixe, adaptatif)
│   └── 📁 utils/            # Utilitaires
│       ├── config.py        # Legacy YAML utils (deprecated)
│       └── evaluation.py    # Métriques et évaluation
│
├── 📁 tests/                # Tests unitaires et intégration (NEW)
│   ├── test_rl_config_pydantic.py # Tests config Pydantic
│   ├── test_env_direct_v2_integration.py # Tests environnement
│   └── test_full_episode_training.py # Tests training DQN
│
├── 📁 benchmarks/           # Performance benchmarks (NEW)
│   └── benchmark_env_performance.py # Latency & throughput tests
│
├── 📁 data/                 # Données topologie
│   └── victoria_island_topology.csv # Network topology
│
├── requirements.txt         # Dependencies (Pydantic, NetworkX, etc.)
└── README.md               # Ce fichier
```

**Breaking Changes (2025)**:
- ❌ Removed: `src/endpoint/` (HTTP client obsolete)
- ❌ Removed: `configs/*.yaml` (YAML configuration obsolete)
- ✅ Added: `src/config/` (Pydantic factory)
- ✅ Added: `TrafficSignalEnvDirectV2` (modern environment)
│   ├── traffic_lagos.yaml  # Paramètres trafic Lagos
│   └── lagos_master.yaml   # Configuration maître Lagos
│
├── 📁 data/                # Données réelles
│   ├── donnees_vitesse_historique.csv    # Données vitesses
│   └── fichier_de_travail_corridor.csv   # Corridor Victoria Island
│
├── 📁 scripts/             # Scripts utilitaires
│   ├── demo.py            # Démonstrations interactives
│   ├── train.py           # Script entraînement principal
│   ├── analyze_corridor.py # Analyse données corridor
│   ├── adapt_lagos.py     # Génération configs Lagos
│   └── test_lagos.py      # Tests configuration Lagos
│
├── 📁 tests/              # Tests unitaires
│   └── test_components.py # Tests composants système
│
└── 📁 docs/              # Documentation
    ├── plan_code.md      # Architecture détaillée
    └── implementation/   # Documentation technique
```

## 🔧 Installation et Configuration

### Prérequis Système
- **Python 3.9+** 
- **GPU NVIDIA** with CUDA Compute Capability 6.0+ (required for best performance)
- **CUDA Toolkit 11.x or 12.x**
- **RAM** : 8GB minimum, 16GB recommandé
- **GPU Memory**: 4GB+ recommended
- **Stockage** : 5GB d'espace libre

### Installation des Dépendances

```bash
# Cloner le projet
git clone [URL_REPO]
cd Code_RL

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances Principales
```
# Core RL
gymnasium>=0.28.0
stable-baselines3>=2.0.0

# Configuration (NEW - Pydantic-based)
pydantic>=2.0.0
networkx>=3.0

# GPU acceleration (required)
cupy-cuda11x>=12.0.0  # Match your CUDA version
numba>=0.56.0

# Scientific computing
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
tensorboard>=2.13.0

# Development
pytest>=7.3.0
black>=23.3.0
mypy>=1.3.0
```

## 🚀 Utilisation

### 1. Configuration Pydantic (Modern Approach)

```python
from Code_RL.src.config import create_rl_training_config
from Code_RL.src.env.traffic_signal_env_direct_v2 import TrafficSignalEnvDirectV2

# Create configuration from topology CSV
config = create_rl_training_config(
    csv_topology_path='data/victoria_island_topology.csv',
    episode_duration=3600.0,  # 1 hour episodes
    decision_interval=15.0,   # RL decision every 15s
    default_density=25.0,     # Initial traffic density (veh/km)
    quiet=False
)

# Create environment
env = TrafficSignalEnvDirectV2(
    simulation_config=config,
    quiet=False
)

# Test environment
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action=0)
```

### 2. Entraînement DQN avec Stable-Baselines3

```python
from stable_baselines3 import DQN
from Code_RL.src.config import create_rl_training_config
from Code_RL.src.env.traffic_signal_env_direct_v2 import TrafficSignalEnvDirectV2

# Create environment
config = create_rl_training_config(
    csv_topology_path='data/victoria_island_topology.csv',
    episode_duration=1800.0,
    decision_interval=15.0
)
env = TrafficSignalEnvDirectV2(simulation_config=config)

# Create DQN agent
model = DQN(
    'MlpPolicy',
    env,
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    tensorboard_log='./logs/dqn_traffic/'
)

# Train
model.learn(total_timesteps=100000)
model.save('dqn_traffic_control')

# Evaluate
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### 3. Tests et Benchmarks

```bash
# Run unit tests
pytest Code_RL/tests/ -v

# Run performance benchmarks
python Code_RL/benchmarks/benchmark_env_performance.py

# Expected output:
# Step latency: ~0.2-0.6ms
# Throughput: ~1000+ steps/sec
```

# Génération configuration Lagos
python adapt_lagos.py
```

## ⚙️ Configuration Lagos

Le système utilise une configuration spécialement adaptée au contexte de Victoria Island :

### Configuration Trafic (`traffic_lagos.yaml`)
```yaml
traffic:
  context: "Victoria Island Lagos"
  vehicle_composition:
    motorcycles: 0.35    # 35% motos
    cars: 0.65          # 65% voitures
  
  # Paramètres motos
  motorcycles:
    v_free: 32.0        # km/h vitesse libre
    rho_max: 250        # véh/km densité max
    
  # Paramètres voitures  
  cars:
    v_free: 28.0        # km/h vitesse libre
    rho_max: 120        # véh/km densité max
```

### Configuration Environnement (`env_lagos.yaml`)
```yaml
environment:
  dt_decision: 10.0     # Décisions toutes les 10s
  
  reward:
    w_wait_time: 1.2    # Poids temps attente (élevé)
    w_queue_length: 0.6 # Poids longueur files
    w_throughput: 1.0   # Poids débit
    w_switch_penalty: 0.1 # Pénalité changements
```

### Configuration Signalisation (`signals_lagos.yaml`)
```yaml
signals:
  timings:
    min_green: 15.0     # Vert minimum 15s (trafic dense)
    max_green: 90.0     # Vert maximum 90s
    yellow: 4.0         # Jaune 4s (sécurité piétons)
    all_red: 3.0        # Rouge général 3s
```

## 📊 Réseau Victoria Island

Le système modélise 2 intersections clés du corridor Victoria Island :

### Intersection 1 - Nœud 2339926113
- **Nord-Sud** : Akin Adesola Street (primary, 3 voies, 50 km/h)
- **Est-Ouest** : Adeola Odeku Street (secondary, 2 voies, 40 km/h)

### Intersection 2 - Nœud 95636900  
- **Nord-Sud** : Akin Adesola Street (primary, 3 voies, 50 km/h)
- **Est-Ouest** : Adeola Odeku Street (secondary, 2 voies, 40 km/h)

### 8 Branches de Trafic
```
intersection_1_north_in   -> Entrée Nord Intersection 1
intersection_1_south_in   -> Entrée Sud Intersection 1  
intersection_1_north_out  -> Sortie Nord Intersection 1
intersection_1_south_out  -> Sortie Sud Intersection 1
intersection_2_north_in   -> Entrée Nord Intersection 2
intersection_2_south_in   -> Entrée Sud Intersection 2
intersection_2_north_out  -> Sortie Nord Intersection 2
intersection_2_south_out  -> Sortie Sud Intersection 2
```

## 📈 Métriques et Évaluation

### Métriques Principales
- **Temps d'attente moyen** : Temps véhicules à l'arrêt
- **Longueur des files** : Nombre véhicules en attente
- **Débit** : Véhicules/heure traversant l'intersection
- **Nombre de changements** : Fréquence commutations phases

### Comparaison Performance
- **Agent DQN** vs **Baseline fixe** (cycles fixes)
- **Évaluation** : 10+ épisodes avec graines aléatoires
- **Stabilité** : Variance des performances

## 🧪 Tests et Validation

### Tests Unitaires
```bash
pytest tests/test_components.py -v
```

### Tests d'Intégration
```bash
python test_lagos.py
```

### Validation des Configurations
```bash
python validate.py
```

## 📋 Données d'Entrée

### Format des Données Corridor
Le fichier `fichier_de_travail_corridor.csv` contient :
- **Node_from/Node_to** : Identifiants nœuds intersection
- **Street_name** : Nom de rue
- **Highway** : Type de route (primary/secondary/tertiary)
- **Oneway** : Direction (yes/no)
- **Length_m** : Longueur segment en mètres

### Analyse Automatique
Le script `analyze_corridor.py` :
1. **Identifie** les intersections majeures
2. **Génère** la topologie réseau
3. **Crée** le fichier `network_real.yaml`
4. **Configure** les paramètres par type de route

## 🔄 Processus de Développement

### 1. Phase d'Analyse
- Analyse données corridor Victoria Island
- Identification intersections clés
- Caractérisation types de trafic

### 2. Phase d'Adaptation
- Création configurations spécifiques Lagos
- Calibrage paramètres trafic
- Ajustement fonction de récompense

### 3. Phase de Test
- Validation composants individuels
- Tests intégration complète
- Comparaison avec baselines

### 4. Phase d'Évaluation
- Entraînement modèles DQN
- Évaluation performances
- Analyse stabilité

## 🐛 Débogage et Diagnostic

### Logs et Diagnostics
```bash
# Logs détaillés pendant entraînement
python train.py --config lagos --use-mock --timesteps 1000 --verbose

# Test composants individuels
python test_lagos.py

# Validation configuration
python -c "from utils.config import load_config; print(load_config('configs/env_lagos.yaml'))"
```

### Problèmes Courants

1. **Erreur import modules**
   ```bash
   # Vérifier PYTHONPATH
   export PYTHONPATH="${PYTHONPATH}:./src"
   ```

2. **Configuration manquante**
   ```bash
   # Regénérer configs Lagos
   python adapt_lagos.py
   ```

3. **Erreur réseau réel**
   ```bash
   # Regénérer réseau Victoria Island
   python analyze_corridor.py
   ```

## 📊 Résultats Expérimentaux

### Performance Baseline
```
Agent DQN Lagos:
- Récompense moyenne: -0.01 ± 0.00
- Changements de phase: 90/épisode
- Convergence: ~1000 timesteps

Baseline Fixe:
- Changements de phase: 59/épisode
- Cycles fixes 60s/60s
```

### Observations
- **Agent adaptatif** : Plus de changements de phase (réactivité)
- **Timing Lagos** : Respect contraintes 15s-90s
- **Stabilité** : Faible variance sur 10 épisodes

## 🔮 Extensions Futures

### Intégrations Possibles
1. **SUMO** : Simulation trafic réaliste
2. **CARLA** : Environnement 3D avec véhicules autonomes
3. **Real-time data** : APIs trafic temps réel Lagos

### Améliorations Algorithmiques
1. **Multi-agent** : Coordination plusieurs intersections
2. **A3C/PPO** : Algorithmes plus avancés
3. **Transfer learning** : Adaptation autres villes

### Extensions Réseau
1. **Plus d'intersections** : Corridor complet Victoria Island
2. **Modes de transport** : Piétons, bus, BRT
3. **Optimisation réseau** : Coordination globale

## 📞 Support et Contribution

### Structure du Code
- **Modulaire** : Composants indépendants testables
- **Configurable** : Toutes les configurations externalisées
- **Extensible** : Interfaces claires pour extensions

### Tests de Régression
Avant toute modification majeure :
```bash
# Tests complets
python test_lagos.py
pytest tests/
python demo.py 1
python train.py --config lagos --use-mock --timesteps 100
```

### Documentation
- **Code documenté** : Docstrings Python standard
- **Configuration** : Commentaires YAML explicatifs  
- **Architecture** : Schémas et diagrammes

## 📄 Licence et Citation

Projet développé pour l'optimisation du trafic urbain à Lagos, Nigeria.

### Citation Suggérée
```bibtex
@software{lagos_traffic_rl_2025,
  title={Système de Contrôle de Signalisation par Apprentissage par Renforcement - Victoria Island Lagos},
  author={[Auteur]},
  year={2025},
  url={[URL_REPO]}
}
```

---

**Note** : Ce système est optimisé pour le contexte spécifique de Victoria Island Lagos mais peut être adapté à d'autres environnements urbains en modifiant les configurations appropriées.
