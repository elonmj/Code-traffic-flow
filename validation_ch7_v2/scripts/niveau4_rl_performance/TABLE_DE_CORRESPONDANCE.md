# Table de Correspondance - Ancien Système  Nouvelle Architecture

**Document**: Mapping Complet des Fonctions/Classes
**Objectif**: Traçabilité 100% pour préserver les innovations

---

## Mapping Global

| Ancien Fichier | Ancienne Fonction/Classe | Nouveau Module | Nouvelle Classe/Fonction | Innovation | Statut |
|----------------|-------------------------|----------------|-------------------------|------------|---------|
| test_section_7_6_rl_performance.py | _save_baseline_cache() | domain/cache/cache_manager.py | CacheManager.save_baseline() | Innovation 1 | À IMPL |
| test_section_7_6_rl_performance.py | _load_baseline_cache() | domain/cache/cache_manager.py | CacheManager.load_baseline() | Innovation 1 | À IMPL |
| test_section_7_6_rl_performance.py | _compute_config_hash() | domain/checkpoint/config_hasher.py | ConfigHasher.compute_hash() | Innovation 2 | À IMPL |
| test_section_7_6_rl_performance.py | _save_checkpoint_with_rotation() | domain/checkpoint/checkpoint_manager.py | CheckpointManager.save_with_rotation() | Innovations 2,5 | À IMPL |
| test_section_7_6_rl_performance.py | _load_checkpoint_if_compatible() | domain/checkpoint/checkpoint_manager.py | CheckpointManager.load_if_compatible() | Innovation 2 | À IMPL |
| test_section_7_6_rl_performance.py | _rotate_checkpoints() | domain/checkpoint/checkpoint_manager.py | CheckpointManager._rotate() | Innovation 5 | À IMPL |
| test_section_7_6_rl_performance.py | BaselineController | domain/controllers/baseline_controller.py | BaselineController | Innovation 3,8 | À IMPL |
| test_section_7_6_rl_performance.py | RLController | domain/controllers/rl_controller.py | RLController | Innovation 3 | À IMPL |
| test_section_7_6_rl_performance.py | _setup_dual_logging() | infrastructure/logging/structured_logger.py | StructuredLogger.setup_dual_logging() | Innovation 7 | À IMPL |
| test_section_7_6_rl_performance.py | DQN_HYPERPARAMETERS (import) | config/section_7_6_rl_performance.yaml | rl_algorithms.dqn | Innovation 6 | À IMPL |
| test_section_7_6_rl_performance.py | _create_benin_context_baseline() | config/section_7_6_rl_performance.yaml | baseline.benin_context | Innovation 8 | À IMPL |
| validation_cli.py | main() | entry_points/cli.py | cli.run() | - | À IMPL |
| run_kaggle_validation_section_7_6.py | main() | entry_points/cli.py | cli.run() | - | SUPPRIMÉ |
| validation_utils.py | run_validation_test() | entry_points/cli.py | cli.run() | - | SUPPRIMÉ |

---

## Détail par Innovation

### Innovation 1: Cache Additif Baseline
**Ancien**: test_section_7_6_rl_performance.py (lignes 450-520)
**Nouveau**: domain/cache/cache_manager.py + infrastructure/cache/pickle_storage.py

### Innovation 2: Config-Hashing Checkpoints  
**Ancien**: test_section_7_6_rl_performance.py (lignes 680-750)
**Nouveau**: domain/checkpoint/checkpoint_manager.py + domain/checkpoint/config_hasher.py

### Innovation 3: Controller State Serialization
**Ancien**: test_section_7_6_rl_performance.py (lignes 580-650)
**Nouveau**: domain/controllers/baseline_controller.py + domain/controllers/rl_controller.py

### Innovation 4: Dual Cache System
**Ancien**: Architecture cache/ avec baseline/ et rl/
**Nouveau**: PRÉSERVÉ identiquement dans domain/cache/cache_manager.py

### Innovation 5: Checkpoint Rotation
**Ancien**: test_section_7_6_rl_performance.py (lignes 820-890)
**Nouveau**: domain/checkpoint/checkpoint_manager.py (méthode _rotate)

### Innovation 6: DRY Hyperparameters
**Ancien**: Import depuis Code_RL (lignes 120-180)
**Nouveau**: config/section_7_6_rl_performance.yaml (référence Code_RL)

### Innovation 7: Dual Logging
**Ancien**: test_section_7_6_rl_performance.py (lignes 220-290)
**Nouveau**: infrastructure/logging/structured_logger.py

### Innovation 8: Baseline Béninois Context
**Ancien**: test_section_7_6_rl_performance.py (lignes 1200-1350)
**Nouveau**: config/section_7_6_rl_performance.yaml (section baseline)

