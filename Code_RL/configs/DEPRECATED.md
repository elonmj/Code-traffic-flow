# DEPRECATED: YAML Configuration Files

**Date**: 2025-01-27  
**Status**: DEPRECATED - Migrate to Pydantic configs

## Migration Notice

This directory contains **legacy YAML configuration files** that are being phased out in favor of the new **Pydantic-based configuration system**.

### Why the Change?

The new Pydantic configuration system provides:

- ✅ **Type Safety**: Configurations are validated at construction time
- ✅ **Better IDE Support**: Autocomplete and type hints
- ✅ **Fail Fast**: Invalid configs caught immediately, not during training
- ✅ **One-Liner Creation**: No need to manage multiple YAML files
- ✅ **Programmatic**: Configs are Python objects, easy to manipulate
- ✅ **Version Control Friendly**: Configs defined in code, not files

### How to Migrate

**OLD WAY (YAML - Deprecated)**:
```python
from utils.config import load_configs

# Load 4+ YAML files
configs = load_configs("configs")
env = create_environment(configs)
```

**NEW WAY (Pydantic - Recommended)**:
```python
from utils.config import RLConfigBuilder
from rl.train_dqn import create_environment_pydantic

# One-liner configuration
rl_config = RLConfigBuilder.for_training(
    scenario="lagos",
    N=200,
    episode_length=3600.0
)

env = create_environment_pydantic(rl_config)
```

### Command Line Migration

**OLD**:
```bash
python train.py --config lagos --config-dir configs
```

**NEW**:
```bash
python train.py --use-pydantic --scenario lagos --grid-size 200
```

### File Status

| File | Status | Migration Path |
|------|--------|----------------|
| `endpoint.yaml` | DEPRECATED | Included in RLConfigBuilder.endpoint_params |
| `env.yaml` | DEPRECATED | Included in RLConfigBuilder.rl_env_params |
| `env_lagos.yaml` | DEPRECATED | Use `scenario="lagos"` in RLConfigBuilder |
| `network.yaml` | DEPRECATED | Included in ARZ SimulationConfig |
| `network_real.yaml` | DEPRECATED | Included in ARZ SimulationConfig |
| `signals.yaml` | DEPRECATED | Included in RLConfigBuilder.signal_params |
| `signals_lagos.yaml` | DEPRECATED | Use `scenario="lagos"` in RLConfigBuilder |
| `traffic_lagos.yaml` | DEPRECATED | Params included in `scenario="lagos"` |
| `lagos_master.yaml` | DEPRECATED | Use `scenario="lagos"` in RLConfigBuilder |

### Files Preserved for Reference

These YAML files are **kept for reference only** and will not be loaded by new code. They document the original configuration structure for:

- Historical reference
- Understanding migration mapping
- Backward compatibility testing

### Timeline

- **2025-01-27**: Pydantic system introduced, YAML deprecated
- **Future**: YAML loading will show deprecation warnings
- **Long-term**: YAML support may be removed entirely

### Support

For migration questions, see:
- [`src/utils/config.py`](../src/utils/config.py) - RLConfigBuilder documentation
- [`tests/test_config_migration.py`](../tests/test_config_migration.py) - Migration examples (if available)
- Project README.md - Updated usage examples

### Quick Reference: Configuration Mapping

```python
# YAML → Pydantic mapping
YAML_configs = {
    "endpoint": {
        "protocol": "mock",       # → rl_config.endpoint_params["protocol"]
        "dt_sim": 0.5,           # → rl_config.arz_simulation_config.dt
    },
    "env": {
        "dt_decision": 15.0,     # → rl_config.rl_env_params["dt_decision"]
        "episode_length": 3600,  # → rl_config.rl_env_params["episode_length"]
        "normalization": {...},  # → rl_config.rl_env_params["normalization"]
        "reward": {...},         # → rl_config.rl_env_params["reward"]
    },
    "signals": {
        "phases": [...],         # → rl_config.signal_params["signals"]["phases"]
    },
    "network": {
        "branches": [...],       # → Extracted from arz_simulation_config
    }
}

# Create equivalent Pydantic config
rl_config = RLConfigBuilder.for_training(
    scenario="lagos",  # Auto-configures all the above
    N=200,
    episode_length=3600.0,
    dt_decision=15.0
)
```

---

**Do NOT edit these YAML files**. They are preserved for reference only.  
Use `RLConfigBuilder` for all new development.
