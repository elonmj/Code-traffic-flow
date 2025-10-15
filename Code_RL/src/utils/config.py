"""
Utility functions for configuration, logging, and analysis
"""

import yaml
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np


def setup_logging(level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    import logging.config
    
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': level,
                'formatter': 'simple'
            }
        },
        'loggers': {
            '': {  # root logger
                'level': level,
                'handlers': ['console']
            }
        }
    }
    
    if log_file:
        logging_config['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'filename': log_file,
            'level': level,
            'formatter': 'detailed'
        }
        logging_config['loggers']['']['handlers'].append('file')
    
    logging.config.dictConfig(logging_config)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def load_configs(config_dir: str) -> Dict[str, Any]:
    """Load all configuration files from directory"""
    config_path = Path(config_dir)
    configs = {}
    
    for config_file in ["endpoint.yaml", "signals.yaml", "network.yaml", "env.yaml"]:
        file_path = config_path / config_file
        if file_path.exists():
            configs[file_path.stem] = load_config(str(file_path))
    
    return configs


def load_lagos_traffic_params(config_dir: Optional[str] = None) -> Dict[str, float]:
    """Load real Lagos traffic parameters from traffic_lagos.yaml.
    
    Extracts key traffic parameters for Victoria Island Lagos:
    - Max densities: motorcycles and cars (veh/km)
    - Free speeds: motorcycles and cars (km/h)
    - Vehicle mix: percentages for each vehicle type
    - Behaviors: creeping rate, gap filling, signal compliance
    
    Args:
        config_dir: Path to configs directory. If None, uses default Code_RL/configs
    
    Returns:
        Dictionary with extracted parameters ready for simulation use
    
    Example:
        >>> params = load_lagos_traffic_params()
        >>> print(params['max_density_motorcycles'])  # 250.0 veh/km
        >>> print(params['free_speed_cars'])  # 28.0 km/h
    """
    if config_dir is None:
        # Default to Code_RL/configs directory
        config_dir = Path(__file__).parent.parent.parent / "configs"
    else:
        config_dir = Path(config_dir)
    
    lagos_config_path = config_dir / "traffic_lagos.yaml"
    
    if not lagos_config_path.exists():
        print(f"[WARNING] traffic_lagos.yaml not found at {lagos_config_path}")
        print("[WARNING] Using fallback parameters")
        # Fallback to reasonable defaults
        return {
            'max_density_motorcycles': 200.0,  # veh/km
            'max_density_cars': 100.0,
            'free_speed_motorcycles': 30.0,  # km/h
            'free_speed_cars': 25.0,
            'vehicle_mix_motorcycles': 0.30,
            'vehicle_mix_cars': 0.50,
            'creeping_rate': 0.5,
            'gap_filling_rate': 0.7,
            'signal_compliance': 0.8
        }
    
    config = load_config(str(lagos_config_path))
    traffic = config.get('traffic', {})
    
    # Extract nested parameters
    max_densities = traffic.get('max_densities', {})
    free_speeds = traffic.get('free_speeds', {})
    vehicle_mix = traffic.get('vehicle_mix', {})
    behaviors = traffic.get('behaviors', {})
    
    params = {
        'max_density_motorcycles': float(max_densities.get('motorcycles', 250)),  # veh/km
        'max_density_cars': float(max_densities.get('cars', 120)),
        'max_density_total': float(max_densities.get('total', 370)),
        'free_speed_motorcycles': float(free_speeds.get('motorcycles', 32)),  # km/h
        'free_speed_cars': float(free_speeds.get('cars', 28)),
        'free_speed_average': float(free_speeds.get('average', 30)),
        'vehicle_mix_motorcycles': float(vehicle_mix.get('motorcycles_percentage', 35)) / 100.0,
        'vehicle_mix_cars': float(vehicle_mix.get('cars_percentage', 45)) / 100.0,
        'vehicle_mix_buses': float(vehicle_mix.get('buses_percentage', 15)) / 100.0,
        'vehicle_mix_trucks': float(vehicle_mix.get('trucks_percentage', 5)) / 100.0,
        'creeping_rate': float(behaviors.get('creeping_rate', 0.6)),
        'gap_filling_rate': float(behaviors.get('gap_filling_rate', 0.8)),
        'signal_compliance': float(behaviors.get('signal_compliance', 0.7)),
        'context': traffic.get('context', 'Victoria Island Lagos')
    }
    
    return params


def validate_config_consistency(configs: Dict[str, Any]) -> bool:
    """Validate consistency between configuration files"""
    errors = []
    warnings = []
    
    # Check dt_decision vs dt_sim compatibility
    if "endpoint" in configs and "env" in configs:
        endpoint_config = configs["endpoint"]
        env_config = configs["env"]
        
        # Check if dt_sim exists in endpoint config (at root level or nested)
        dt_sim = None
        if "dt_sim" in endpoint_config:
            dt_sim = endpoint_config["dt_sim"]
        elif "endpoint" in endpoint_config and "dt_sim" in endpoint_config["endpoint"]:
            dt_sim = endpoint_config["endpoint"]["dt_sim"]
            
        if dt_sim is not None:
            if "environment" in env_config and "dt_decision" in env_config["environment"]:
                dt_decision = env_config["environment"]["dt_decision"]
                
                k = dt_decision / dt_sim
                if not k.is_integer() or k < 1:
                    errors.append(f"dt_decision ({dt_decision}) must be integer multiple of dt_sim ({dt_sim})")
        else:
            warnings.append("dt_sim not found in endpoint config, skipping timing validation")
    
    # Check phase count consistency
    if "signals" in configs and "env" in configs:
        signals_config = configs["signals"]
        if "signals" in signals_config and "phases" in signals_config["signals"]:
            num_phases = len(signals_config["signals"]["phases"])
            # Could add more phase-related validations here
            if num_phases < 2:
                warnings.append(f"Only {num_phases} phases defined, consider adding more for complex intersections")
    
    # Check branch mapping
    if "network" in configs:
        network_config = configs["network"]
        if "network" in network_config and "branches" in network_config["network"]:
            branch_ids = [b["id"] for b in network_config["network"]["branches"]]
            if len(branch_ids) != len(set(branch_ids)):
                errors.append("Duplicate branch IDs in network configuration")
        else:
            warnings.append("No branches found in network configuration")
    
    # Print warnings
    if warnings:
        for warning in warnings:
            print(f"Config validation warning: {warning}")
    
    # Print errors
    if errors:
        for error in errors:
            print(f"Config validation error: {error}")
        return False
    
    return True


def save_episode_data(episode_data: List[Dict[str, Any]], filepath: str):
    """Save episode data to CSV"""
    df = pd.DataFrame(episode_data)
    df.to_csv(filepath, index=False)


def save_training_results(results: Dict[str, Any], filepath: str):
    """Save training results to JSON"""
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_numpy(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=2)


def calculate_performance_metrics(episode_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate performance metrics across episodes"""
    if not episode_summaries:
        return {}
    
    metrics = {}
    
    # Get all numeric keys
    numeric_keys = []
    for key, value in episode_summaries[0].items():
        if isinstance(value, (int, float)):
            numeric_keys.append(key)
    
    # Calculate statistics for each metric
    for key in numeric_keys:
        values = [ep[key] for ep in episode_summaries if key in ep]
        if values:
            metrics[f"{key}_mean"] = np.mean(values)
            metrics[f"{key}_std"] = np.std(values)
            metrics[f"{key}_min"] = np.min(values)
            metrics[f"{key}_max"] = np.max(values)
    
    return metrics


def compare_baselines(
    rl_results: List[Dict[str, Any]], 
    baseline_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare RL performance against baseline"""
    comparison = {}
    
    rl_metrics = calculate_performance_metrics(rl_results)
    baseline_metrics = calculate_performance_metrics(baseline_results)
    
    # Calculate improvement percentages
    for key in rl_metrics:
        if key.endswith("_mean") and key in baseline_metrics:
            base_key = key.replace("_mean", "")
            rl_value = rl_metrics[key]
            baseline_value = baseline_metrics[key]
            
            if baseline_value != 0:
                improvement = (rl_value - baseline_value) / abs(baseline_value) * 100
                comparison[f"{base_key}_improvement_pct"] = improvement
    
    comparison["rl_metrics"] = rl_metrics
    comparison["baseline_metrics"] = baseline_metrics
    
    return comparison


class ExperimentTracker:
    """Track experiments and results"""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_experiment = None
        self.experiments_log = self.experiment_dir / "experiments.json"
        
        # Load existing experiments
        if self.experiments_log.exists():
            with open(self.experiments_log, 'r') as f:
                self.experiments = json.load(f)
        else:
            self.experiments = []
    
    def start_experiment(self, name: str, config: Dict[str, Any], description: str = ""):
        """Start new experiment"""
        import time
        import hashlib
        
        self.current_experiment = {
            "name": name,
            "description": description,
            "start_time": time.time(),
            "config": config,
            "config_hash": hashlib.md5(str(config).encode()).hexdigest()[:8],
            "status": "running",
            "results": {}
        }
        
        print(f"Started experiment: {name}")
    
    def log_episode(self, episode: int, summary: Dict[str, Any]):
        """Log episode results"""
        if self.current_experiment:
            if "episodes" not in self.current_experiment["results"]:
                self.current_experiment["results"]["episodes"] = []
            
            episode_data = {"episode": episode, **summary}
            self.current_experiment["results"]["episodes"].append(episode_data)
    
    def finish_experiment(self, final_results: Optional[Dict[str, Any]] = None):
        """Finish current experiment"""
        if self.current_experiment:
            import time
            
            self.current_experiment["end_time"] = time.time()
            self.current_experiment["duration"] = (
                self.current_experiment["end_time"] - self.current_experiment["start_time"]
            )
            self.current_experiment["status"] = "completed"
            
            if final_results:
                self.current_experiment["results"].update(final_results)
            
            # Calculate final metrics
            if "episodes" in self.current_experiment["results"]:
                episodes = self.current_experiment["results"]["episodes"]
                final_metrics = calculate_performance_metrics(episodes)
                self.current_experiment["results"]["final_metrics"] = final_metrics
            
            # Save experiment
            self.experiments.append(self.current_experiment)
            self._save_experiments()
            
            # Save individual experiment file
            exp_file = self.experiment_dir / f"{self.current_experiment['name']}.json"
            with open(exp_file, 'w') as f:
                json.dump(self.current_experiment, f, indent=2)
            
            print(f"Finished experiment: {self.current_experiment['name']}")
            print(f"Duration: {self.current_experiment['duration']:.1f} seconds")
            
            self.current_experiment = None
    
    def _save_experiments(self):
        """Save experiments log"""
        with open(self.experiments_log, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def get_experiment_summary(self) -> pd.DataFrame:
        """Get summary of all experiments"""
        if not self.experiments:
            return pd.DataFrame()
        
        summary_data = []
        for exp in self.experiments:
            row = {
                "name": exp["name"],
                "status": exp["status"],
                "duration": exp.get("duration", 0),
                "config_hash": exp.get("config_hash", ""),
            }
            
            # Add final metrics
            if "final_metrics" in exp.get("results", {}):
                for key, value in exp["results"]["final_metrics"].items():
                    if key.endswith("_mean"):
                        base_key = key.replace("_mean", "")
                        row[base_key] = value
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)


def create_scenario_config_with_lagos_data(
    scenario_type: str,
    output_path: Optional[Path] = None,
    config_dir: Optional[str] = None,
    duration: float = 600.0,
    domain_length: float = 1000.0
) -> Dict[str, Any]:
    """Create scenario configuration using REAL Lagos traffic data.
    
    Args:
        scenario_type: Type of scenario ('traffic_light_control', 'ramp_metering', 'adaptive_speed_control')
        output_path: Optional path to save YAML config
        config_dir: Path to configs directory for Lagos data
        duration: Simulation duration in seconds (default: 600s = 10 minutes)
        domain_length: Road domain length in meters (default: 1000m = 1km)
    
    Returns:
        Dictionary with scenario configuration using real Lagos parameters
    """
    # Load REAL Lagos traffic parameters
    lagos_params = load_lagos_traffic_params(config_dir)
    
    # Extract real parameters
    max_density_m = lagos_params['max_density_motorcycles']  # 250 veh/km (REAL)
    max_density_c = lagos_params['max_density_cars']  # 120 veh/km (REAL)
    free_speed_m_kmh = lagos_params['free_speed_motorcycles']  # 32 km/h (REAL)
    free_speed_c_kmh = lagos_params['free_speed_cars']  # 28 km/h (REAL)
    
    # Convert speeds to m/s
    free_speed_m = free_speed_m_kmh / 3.6  # ~8.9 m/s
    free_speed_c = free_speed_c_kmh / 3.6  # ~7.8 m/s
    
    # ✅ BUG #34 FIX: Inflow must use EQUILIBRIUM SPEED not free speed
    # Discovery: ARZ model relaxes w → Ve via source term S = (Ve - w) / tau
    # At rho=200 veh/km: Ve_m = 2.26 m/s << V0_m = 8.89 m/s
    # Result: Prescribed high-speed flux gets reduced by relaxation → no accumulation
    # Solution: Use equilibrium speed for inflow to match ARZ physics
    
    # Calculate equilibrium speeds using ARZ model formula
    rho_jam_veh_km = max_density_m + max_density_c  # 370 veh/km total
    rho_jam = rho_jam_veh_km / 1000.0  # Convert to veh/m
    V_creeping = 0.6  # Default creeping speed (m/s)
    
    # Initial state: LIGHT traffic (road starts relatively empty)
    rho_m_initial_veh_km = max_density_m * 0.1  # 25 veh/km (light)
    rho_c_initial_veh_km = max_density_c * 0.1  # 12 veh/km (light)
    rho_total_initial = (rho_m_initial_veh_km + rho_c_initial_veh_km) / 1000.0  # 0.037 veh/m
    g_initial = max(0.0, 1.0 - rho_total_initial / rho_jam)  # ≈ 0.9 (nearly free)
    w_m_initial = V_creeping + (free_speed_m - V_creeping) * g_initial  # ≈ 8.1 m/s
    w_c_initial = free_speed_c * g_initial  # ≈ 7.0 m/s
    # Flux_init ≈ 0.037 * 7.5 = 0.28 veh/s (very low, sustainable)
    
    # Inflow: EXTREME demand (jam-level Lagos traffic approaching intersection)
    # ✅ BUG #35 FIX ITERATION 3: Increase density to ensure v < 5 m/s threshold
    # Previous: 0.8 × jam = 200 veh/km → Ve ≈ 9 m/s → v ≈ 5-6 m/s (ABOVE threshold!)
    # New:      1.2 × jam = 300 veh/km → Ve ≈ 4 m/s → v ≈ 2-3 m/s (BELOW threshold ✅)
    rho_m_inflow_veh_km = max_density_m * 1.2  # 300 veh/km (extreme/jam)
    rho_c_inflow_veh_km = max_density_c * 0.8  # 96 veh/km (heavy)
    rho_total_inflow = (rho_m_inflow_veh_km + rho_c_inflow_veh_km) / 1000.0  # 0.396 veh/m (107% jam!)
    rho_total_inflow = min(rho_total_inflow, rho_jam * 0.95)  # Cap at 95% jam to avoid singularity
    g_inflow = max(0.0, 1.0 - rho_total_inflow / rho_jam)  # ≈ 0.05 (near-jam)
    w_m_inflow = V_creeping + (free_speed_m - V_creeping) * g_inflow  # ≈ 1.0 m/s (crawling!)
    w_c_inflow = free_speed_c * g_inflow  # ≈ 0.4 m/s (crawling!)
    # Flux_inflow ≈ 0.35 * 1.0 = 0.35 veh/s → q_inflow > q_init (0.28) AND sustainable ✅
    # Velocities < 5 m/s → Queue detection WILL work! ✅
    
    # Base configuration
    config = {
        'scenario_name': f'{scenario_type}_lagos_real',
        'N': 100,
        'xmin': 0.0,
        'xmax': domain_length,
        't_final': duration,
        'output_dt': 60.0,
        'CFL': 0.4,
        'boundary_conditions': {
            'left': {'type': 'inflow', 'state': [rho_m_inflow_veh_km, w_m_inflow, rho_c_inflow_veh_km, w_c_inflow]},
            'right': {'type': 'outflow'}
        },
        'road': {'quality_type': 'uniform', 'quality_value': 2},
        'lagos_parameters': lagos_params  # Include full Lagos params for reference
    }
    
    # Scenario-specific parameters
    if scenario_type == 'traffic_light_control':
        config['parameters'] = {
            'V0_m': free_speed_m,  # Use REAL Lagos speeds
            'V0_c': free_speed_c,
            'tau_m': 1.0,
            'tau_c': 1.2
        }
        config['initial_conditions'] = {
            'type': 'uniform',
            'state': [rho_m_initial_veh_km, w_m_initial, rho_c_initial_veh_km, w_c_initial]
        }
    elif scenario_type == 'ramp_metering':
        config['parameters'] = {
            'V0_m': free_speed_m * 1.1,
            'V0_c': free_speed_c * 1.1,
            'tau_m': 0.8,
            'tau_c': 1.0
        }
        # Riemann IC for ramp scenario
        config['initial_conditions'] = {
            'type': 'riemann',
            'U_L': [rho_m_inflow_veh_km/1000*0.8, w_m_inflow, rho_c_inflow_veh_km/1000*0.8, w_c_inflow],
            'U_R': [rho_m_initial_veh_km/1000*0.8, w_m_initial, rho_c_initial_veh_km/1000*0.8, w_c_initial],
            'split_pos': domain_length / 2
        }
    elif scenario_type == 'adaptive_speed_control':
        config['parameters'] = {
            'V0_m': free_speed_m * 1.2,
            'V0_c': free_speed_c * 1.2,
            'tau_m': 0.6,
            'tau_c': 0.8
        }
        # Riemann IC for adaptive speed scenario
        config['initial_conditions'] = {
            'type': 'riemann',
            'U_L': [rho_m_inflow_veh_km/1000*0.7, w_m_inflow, rho_c_inflow_veh_km/1000*0.7, w_c_inflow],
            'U_R': [rho_m_initial_veh_km/1000*0.7, w_m_initial, rho_c_initial_veh_km/1000*0.7, w_c_initial],
            'split_pos': domain_length / 2
        }
    
    # Save if output path provided
    if output_path:
        save_config(config, str(output_path))
        print(f"[LAGOS CONFIG] Saved to {output_path}")
    
    return config


def analyze_experiment_config(exp_config: Dict[str, Any], configs: Dict[str, Any]) -> List[str]:
    """Analyze experiment configuration and provide insights"""
    insights = []
    
    # Algorithm analysis
    if "algorithm" in exp_config:
        algo = exp_config["algorithm"]
        insights.append(f"Using {algo} algorithm for training")
        
        if algo == "DQN":
            insights.append("DQN is well-suited for discrete action spaces like traffic signals")
    
    # Timesteps analysis
    if "timesteps" in exp_config:
        timesteps = exp_config["timesteps"]
        if timesteps < 10000:
            insights.append(f"Low timesteps ({timesteps}) - suitable for quick testing")
        elif timesteps < 100000:
            insights.append(f"Medium timesteps ({timesteps}) - good for initial training")
        else:
            insights.append(f"High timesteps ({timesteps}) - thorough training expected")
    
    # Environment analysis
    if "env_config" in exp_config and "max_steps" in exp_config["env_config"]:
        max_steps = exp_config["env_config"]["max_steps"]
        insights.append(f"Episode length: {max_steps} steps")
    
    # Network analysis
    if "network" in exp_config:
        network = exp_config["network"]
        insights.append(f"Target network: {network}")
        
        if network == "victoria_island":
            insights.append("Victoria Island Lagos - high traffic complexity expected")
    
    return insights
