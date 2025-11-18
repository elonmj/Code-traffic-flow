"""
Sanity Checker for RL Training

Vérifie AUTOMATIQUEMENT les 5 BUGS MORTELS avant l'entraînement:
- BUG #37: Action truncation (int vs round)
- BUG #33: Traffic flux mismatch (queue toujours zéro)
- BUG #27: Control interval timing
- BUG #36: GPU validation & traffic signal coupling
- BUG Reward: Reward function issues

Source: Code_RL/docs/RL_TRAINING_SURVIVAL_GUIDE.md

PRINCIPE: "Fail fast" - mieux vaut arrêter avant l'entraînement que perdre 4h
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

# Imports locaux
from Code_RL.src.utils.config import RLConfigBuilder
from ..config import SanityCheckConfig

logger = logging.getLogger(__name__)


@dataclass
class SanityCheckResult:
    """Résultat d'un test de sanité"""
    name: str
    passed: bool
    message: str
    details: Optional[Dict] = None
    
    def __repr__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} {self.name}: {self.message}"


class SanityChecker:
    """
    Vérificateur de sanité pré-entraînement
    
    Usage:
        checker = SanityChecker(rl_config, sanity_config)
        results = checker.run_all_checks()
        if not checker.all_passed(results):
            raise RuntimeError("Sanity checks failed!")
    """
    
    def __init__(
        self,
        rl_config: RLConfigBuilder,
        sanity_config: SanityCheckConfig
    ):
        self.rl_config = rl_config
        self.sanity_config = sanity_config
        self.results: List[SanityCheckResult] = []
    
    def run_all_checks(self) -> List[SanityCheckResult]:
        """
        Exécute TOUS les tests de sanité
        
        Returns:
            Liste des résultats (SanityCheckResult)
        """
        self.results = []
        
        logger.info("=" * 80)
        logger.info("SANITY CHECKS - PRE-TRAINING VALIDATION")
        logger.info("=" * 80)
        
        # Check 1: Action mapping (BUG #37)
        if self.sanity_config.check_action_mapping:
            self.results.append(self._check_action_mapping())
        
        # Check 2: Flux configuration (BUG #33)
        if self.sanity_config.check_flux_config:
            self.results.append(self._check_flux_configuration())
        
        # Check 3: Control interval (BUG #27)
        if self.sanity_config.check_control_interval:
            self.results.append(self._check_control_interval())
        
        # Check 4: Environment rollout (BUG #33, Reward)
        if self.sanity_config.enabled:
            self.results.append(self._check_environment_rollout())
        
        # Check 5: Reward diversity
        if self.sanity_config.enabled:
            self.results.append(self._check_reward_diversity())
        
        # Affichage des résultats
        logger.info("\n" + "=" * 80)
        logger.info("SANITY CHECK RESULTS")
        logger.info("=" * 80)
        
        for result in self.results:
            logger.info(result)
            if result.details:
                for key, value in result.details.items():
                    logger.info(f"  {key}: {value}")
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        logger.info("=" * 80)
        logger.info(f"PASSED: {passed}/{total}")
        
        if passed == total:
            logger.info("✅ ALL CHECKS PASSED - Safe to train!")
        else:
            logger.error("❌ SOME CHECKS FAILED - DO NOT TRAIN!")
        
        logger.info("=" * 80)
        
        return self.results
    
    def all_passed(self, results: Optional[List[SanityCheckResult]] = None) -> bool:
        """Vérifie si tous les tests ont réussi"""
        if results is None:
            results = self.results
        return all(r.passed for r in results)
    
    # =========================================================================
    # INDIVIDUAL CHECKS
    # =========================================================================
    
    def _check_action_mapping(self) -> SanityCheckResult:
        """
        BUG #37: Vérifie que l'action mapping utilise round() et pas int()
        
        SYMPTÔME: Action 1.99 → int() = 1 (mauvais), round() = 2 (correct)
        FIX: Utiliser round(float(action)) dans env.step()
        
        Source: BUG_37_ACTION_TRUNCATION_FIX.md
        """
        logger.info("\n[CHECK 1/5] Action Mapping (BUG #37)")
        
        try:
            # Créer l'environnement
            env = self._create_test_env()
            
            # Test avec action proche de 2.0
            test_action = 1.99
            
            # Vérifier le code de env.step()
            # ATTENTION: On ne peut pas inspecter directement, on teste le comportement
            
            # Simuler ce qui devrait se passer
            int_result = int(test_action)  # = 1 (mauvais)
            round_result = round(float(test_action))  # = 2 (correct)
            
            # On vérifie que l'environnement a bien le fix
            # Pour cela, on checke le code source ou on teste le comportement
            
            # HEURISTIQUE: Si le code contient "round(", c'est OK
            import inspect
            env_code = inspect.getsource(env.step)
            
            has_round = "round(" in env_code
            has_int = "int(" in env_code and "round(" not in env_code
            
            if has_round:
                return SanityCheckResult(
                    name="Action Mapping",
                    passed=True,
                    message="✓ Uses round() for action conversion (BUG #37 fixed)",
                    details={
                        "test_action": test_action,
                        "int_result": int_result,
                        "round_result": round_result,
                        "method": "round()"
                    }
                )
            elif has_int:
                return SanityCheckResult(
                    name="Action Mapping",
                    passed=False,
                    message="✗ Uses int() instead of round() - BUG #37 present!",
                    details={
                        "test_action": test_action,
                        "int_result": int_result,
                        "expected": round_result,
                        "fix_required": "Change int(action) to round(float(action))"
                    }
                )
            else:
                return SanityCheckResult(
                    name="Action Mapping",
                    passed=False,
                    message="⚠ Cannot verify action conversion method",
                    details={"warning": "Manual inspection required"}
                )
        
        except Exception as e:
            return SanityCheckResult(
                name="Action Mapping",
                passed=False,
                message=f"Error during check: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_flux_configuration(self) -> SanityCheckResult:
        """
        BUG #33: Vérifie que rho_inflow >> rho_initial (congestion possible)
        
        SYMPTÔME: Queue toujours zéro, pas de congestion
        FIX: rho_inflow = 0.15, rho_initial = 0.01 (ratio 15:1)
        
        Source: BUG_33_TRAFFIC_FLUX_MISMATCH_ANALYSIS.md
        """
        logger.info("\n[CHECK 2/5] Flux Configuration (BUG #33)")
        
        try:
            # Récupérer la config ARZ
            arz_config = self.rl_config.arz_simulation_config
            
            # Extraire les densités
            # ATTENTION: Dépend de la structure de SimulationConfig
            # On assume qu'il y a des attributs rho_inflow et rho_initial
            
            # Pour l'instant, on vérifie manuellement via les params
            # TODO: Adapter selon la vraie structure de SimulationConfig
            
            # Heuristique: Vérifier que rho_inflow > 10 * rho_initial
            rho_inflow = getattr(arz_config, 'rho_inflow', None)
            rho_initial = getattr(arz_config, 'rho_initial', None)
            
            if rho_inflow is None or rho_initial is None:
                return SanityCheckResult(
                    name="Flux Configuration",
                    passed=False,
                    message="⚠ Cannot extract rho_inflow/rho_initial from config",
                    details={"warning": "Manual verification required"}
                )
            
            ratio = rho_inflow / rho_initial if rho_initial > 0 else 0
            min_ratio = 10.0  # Minimum ratio pour congestion
            
            if ratio >= min_ratio:
                return SanityCheckResult(
                    name="Flux Configuration",
                    passed=True,
                    message=f"✓ Flux ratio OK: {ratio:.1f}:1 (BUG #33 avoided)",
                    details={
                        "rho_inflow": rho_inflow,
                        "rho_initial": rho_initial,
                        "ratio": f"{ratio:.1f}:1",
                        "min_ratio": f"{min_ratio:.1f}:1"
                    }
                )
            else:
                return SanityCheckResult(
                    name="Flux Configuration",
                    passed=False,
                    message=f"✗ Flux ratio too low: {ratio:.1f}:1 < {min_ratio}:1 - BUG #33 risk!",
                    details={
                        "rho_inflow": rho_inflow,
                        "rho_initial": rho_initial,
                        "ratio": f"{ratio:.1f}:1",
                        "min_ratio": f"{min_ratio:.1f}:1",
                        "fix_required": f"Increase rho_inflow or decrease rho_initial"
                    }
                )
        
        except Exception as e:
            return SanityCheckResult(
                name="Flux Configuration",
                passed=False,
                message=f"Error during check: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_control_interval(self) -> SanityCheckResult:
        """
        BUG #27: Vérifie que control_interval = 15s (pas 60s)
        
        SYMPTÔME: Agent n'apprend rien, environnement en steady-state
        FIX: dt_decision = 15.0 secondes
        
        Source: BUG_27_CONTROL_INTERVAL_TIMING.md
        """
        logger.info("\n[CHECK 3/5] Control Interval (BUG #27)")
        
        try:
            # Récupérer dt_decision de la config RL
            rl_params = self.rl_config.rl_env_params
            dt_decision = rl_params.get('dt_decision', None)
            
            if dt_decision is None:
                return SanityCheckResult(
                    name="Control Interval",
                    passed=False,
                    message="⚠ Cannot extract dt_decision from config",
                    details={"warning": "Manual verification required"}
                )
            
            # Valeur attendue: 15.0 secondes
            expected = 15.0
            tolerance = 5.0  # ±5 secondes acceptable
            
            if abs(dt_decision - expected) <= tolerance:
                return SanityCheckResult(
                    name="Control Interval",
                    passed=True,
                    message=f"✓ Control interval OK: {dt_decision}s (BUG #27 avoided)",
                    details={
                        "dt_decision": f"{dt_decision}s",
                        "expected": f"{expected}s",
                        "tolerance": f"±{tolerance}s"
                    }
                )
            else:
                return SanityCheckResult(
                    name="Control Interval",
                    passed=False,
                    message=f"✗ Control interval wrong: {dt_decision}s ≠ {expected}s - BUG #27 risk!",
                    details={
                        "dt_decision": f"{dt_decision}s",
                        "expected": f"{expected}s",
                        "tolerance": f"±{tolerance}s",
                        "fix_required": f"Set dt_decision to {expected}s in RLConfigBuilder"
                    }
                )
        
        except Exception as e:
            return SanityCheckResult(
                name="Control Interval",
                passed=False,
                message=f"Error during check: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_environment_rollout(self) -> SanityCheckResult:
        """
        Vérifie que l'environnement peut faire un rollout sans erreur
        
        Détecte:
        - Erreurs de dimensions
        - Erreurs de couplage GPU/CPU (BUG #36)
        - Erreurs de simulation
        """
        logger.info("\n[CHECK 4/5] Environment Rollout")
        
        try:
            env = self._create_test_env()
            
            # Reset
            obs = env.reset()
            
            # Rollout de num_steps
            num_steps = self.sanity_config.num_steps
            episode_rewards = []
            max_queue = 0.0
            
            for step in range(num_steps):
                # Action aléatoire
                action = env.action_space.sample()
                
                # Step
                obs, reward, done, info = env.step(action)
                episode_rewards.append(reward)
                
                # Extraire queue length si disponible
                if isinstance(info, dict) and 'queue_length' in info:
                    max_queue = max(max_queue, info['queue_length'])
                
                if done:
                    obs = env.reset()
            
            env.close()
            
            # Vérifications
            if len(episode_rewards) == num_steps:
                return SanityCheckResult(
                    name="Environment Rollout",
                    passed=True,
                    message=f"✓ Rollout successful ({num_steps} steps)",
                    details={
                        "steps": num_steps,
                        "mean_reward": f"{np.mean(episode_rewards):.4f}",
                        "std_reward": f"{np.std(episode_rewards):.4f}",
                        "max_queue": f"{max_queue:.2f}"
                    }
                )
            else:
                return SanityCheckResult(
                    name="Environment Rollout",
                    passed=False,
                    message=f"✗ Rollout incomplete: {len(episode_rewards)}/{num_steps} steps",
                    details={
                        "expected_steps": num_steps,
                        "actual_steps": len(episode_rewards)
                    }
                )
        
        except Exception as e:
            return SanityCheckResult(
                name="Environment Rollout",
                passed=False,
                message=f"✗ Rollout failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_reward_diversity(self) -> SanityCheckResult:
        """
        Vérifie que les rewards sont suffisamment divers
        
        SYMPTÔME: Reward toujours identique → pas d'apprentissage
        FIX: Au moins 5 valeurs uniques sur 100 steps
        
        Source: BUG Reward - RL_TRAINING_SURVIVAL_GUIDE.md
        """
        logger.info("\n[CHECK 5/5] Reward Diversity")
        
        try:
            env = self._create_test_env()
            
            # Collecte des rewards
            obs = env.reset()
            rewards = []
            num_steps = self.sanity_config.num_steps
            
            for _ in range(num_steps):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                rewards.append(reward)
                
                if done:
                    obs = env.reset()
            
            env.close()
            
            # Analyse de la diversité
            unique_rewards = len(set(rewards))
            min_unique = self.sanity_config.min_unique_rewards
            
            if unique_rewards >= min_unique:
                return SanityCheckResult(
                    name="Reward Diversity",
                    passed=True,
                    message=f"✓ Reward diversity OK: {unique_rewards} unique values",
                    details={
                        "unique_rewards": unique_rewards,
                        "min_required": min_unique,
                        "mean_reward": f"{np.mean(rewards):.4f}",
                        "std_reward": f"{np.std(rewards):.4f}",
                        "min_reward": f"{np.min(rewards):.4f}",
                        "max_reward": f"{np.max(rewards):.4f}"
                    }
                )
            else:
                return SanityCheckResult(
                    name="Reward Diversity",
                    passed=False,
                    message=f"✗ Reward diversity too low: {unique_rewards} < {min_unique}",
                    details={
                        "unique_rewards": unique_rewards,
                        "min_required": min_unique,
                        "mean_reward": f"{np.mean(rewards):.4f}",
                        "std_reward": f"{np.std(rewards):.4f}",
                        "fix_required": "Check reward function implementation"
                    }
                )
        
        except Exception as e:
            return SanityCheckResult(
                name="Reward Diversity",
                passed=False,
                message=f"Error during check: {str(e)}",
                details={"error": str(e)}
            )
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _create_test_env(self):
        """Crée un environnement de test pour les sanity checks"""
        # Utiliser la factory du RLConfigBuilder
        from Code_RL.src.env.traffic_signal_env_direct import TrafficSignalEnvDirect
        
        env = TrafficSignalEnvDirect(
            simulation_config=self.rl_config.arz_simulation_config,  # Fixed: was arz_simulation_config
            endpoint_params=self.rl_config.endpoint_params,
            signal_params=self.rl_config.signal_params,
            **self.rl_config.rl_env_params
        )
        
        return env


def run_sanity_checks(
    rl_config: RLConfigBuilder,
    sanity_config: SanityCheckConfig
) -> bool:
    """
    Fonction utilitaire pour exécuter les sanity checks
    
    Args:
        rl_config: Configuration RL
        sanity_config: Configuration des tests
    
    Returns:
        True si tous les tests passent, False sinon
    
    Raises:
        RuntimeError si les tests échouent (selon config)
    """
    checker = SanityChecker(rl_config, sanity_config)
    results = checker.run_all_checks()
    
    if not checker.all_passed(results):
        failed = [r for r in results if not r.passed]
        error_msg = "\n".join([f"  - {r.name}: {r.message}" for r in failed])
        raise RuntimeError(
            f"Sanity checks failed ({len(failed)}/{len(results)}):\n{error_msg}"
        )
    
    return True
