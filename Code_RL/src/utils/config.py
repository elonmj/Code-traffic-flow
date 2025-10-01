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
