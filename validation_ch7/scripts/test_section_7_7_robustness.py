#!/usr/bin/env python3
"""
Validation Script: Section 7.7 - Robustness Validation

Tests for Revendication R6: Robustesse sous conditions degradees

This script validates system robustness by:
- Testing stability under extreme conditions
- Validating performance with degraded road capacity R(x)
- Testing numerical stability with low CFL numbers
- Validating behavior at domain boundaries
"""

import sys
import os
import numpy as np
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "code"))

from validation_ch7.scripts.validation_utils import (
    ValidationTest, create_test_config, run_mock_simulation,
    generate_latex_table, save_validation_results
)
from code.analysis.metrics import (
    compute_mape, compute_rmse, calculate_total_mass
)
from code.simulation.runner import SimulationRunner
from code.core.parameters import ModelParameters


class RobustnessValidationTest(ValidationTest):
    """Robustness validation test implementation."""
    
    def __init__(self):
        super().__init__("Robustness Validation", "7.7")
        self.extreme_conditions = {
            'very_low_capacity': {'R_min': 0.1, 'R_variation': 0.3},
            'capacity_bottleneck': {'R_min': 0.2, 'R_variation': 0.8},
            'oscillating_capacity': {'R_min': 0.3, 'R_variation': 0.6},
            'extreme_density': {'rho_max': 0.95, 'initial_density': 0.9},
            'near_vacuum': {'rho_max': 1.0, 'initial_density': 0.01}
        }
    
    def create_degraded_capacity_function(self, condition_type, domain_length=10.0, grid_size=200):
        """Create degraded road capacity function R(x)."""
        x = np.linspace(0, domain_length, grid_size)
        
        if condition_type == 'very_low_capacity':
            # Uniformly low capacity
            R_x = 0.1 + 0.05 * np.sin(2 * np.pi * x / domain_length)
            
        elif condition_type == 'capacity_bottleneck':
            # Severe bottleneck in middle
            bottleneck_center = domain_length / 2
            bottleneck_width = domain_length / 8
            
            R_x = np.ones_like(x) * 0.8
            bottleneck_mask = np.abs(x - bottleneck_center) < bottleneck_width
            R_x[bottleneck_mask] = 0.2 + 0.1 * np.cos(
                np.pi * (x[bottleneck_mask] - bottleneck_center) / bottleneck_width
            )
            
        elif condition_type == 'oscillating_capacity':
            # Rapidly oscillating capacity
            R_x = 0.5 + 0.3 * np.cos(8 * np.pi * x / domain_length) * np.exp(-0.1 * x)
            
        elif condition_type in ['extreme_density', 'near_vacuum']:
            # Normal capacity for density tests
            R_x = np.ones_like(x) * 0.8
            
        else:
            # Default uniform capacity  
            R_x = np.ones_like(x) * 0.7
        
        # Ensure physical bounds
        R_x = np.clip(R_x, 0.05, 1.0)
        
        return R_x
    
    def create_extreme_initial_conditions(self, condition_type, grid_size=200):
        """Create extreme initial conditions for testing."""
        if condition_type == 'extreme_density':
            # Near-jam conditions
            density = np.ones(grid_size) * 0.9
            # Add small random perturbations
            density += np.random.normal(0, 0.02, grid_size)
            density = np.clip(density, 0.85, 0.95)
            
            # Corresponding low velocities
            velocity = 0.05 + 0.1 * (1 - density)
            velocity = np.clip(velocity, 0.02, 0.2)
            
        elif condition_type == 'near_vacuum':
            # Very low density conditions
            density = np.ones(grid_size) * 0.01
            density += np.random.uniform(-0.005, 0.005, grid_size)
            density = np.clip(density, 0.005, 0.02)
            
            # High velocities in free flow
            velocity = np.ones(grid_size) * 0.95
            velocity += np.random.normal(0, 0.05, grid_size)
            velocity = np.clip(velocity, 0.8, 1.0)
            
        else:
            # Default moderate conditions
            density = np.ones(grid_size) * 0.3
            velocity = np.ones(grid_size) * 0.7
        
        return density, velocity
    
    def create_robustness_config(self, condition_type):
        """Create configuration for robustness testing."""
        config = create_test_config(
            grid_size=200,
            domain_length=10.0,
            final_time=3.0,  # Shorter time for stability
            cfl_number=0.3   # Conservative CFL
        )
        
        # Create degraded capacity
        R_x = self.create_degraded_capacity_function(condition_type)
        
        # Modify parameters based on condition
        if condition_type in ['very_low_capacity', 'capacity_bottleneck', 'oscillating_capacity']:
            config['parameters']['V0'] = 0.8
            config['parameters']['tau'] = 0.8
            config['parameters']['rho_max'] = 0.9
            config['road_capacity'] = R_x.tolist()
            
            # Standard initial conditions
            config['initial_conditions'] = {
                'type': 'gaussian_perturbation',
                'base_density': 0.4,
                'perturbation_amplitude': 0.1,
                'perturbation_center': 3.0,
                'perturbation_width': 1.0
            }
            
        elif condition_type in ['extreme_density', 'near_vacuum']:
            config['parameters']['V0'] = 0.9
            config['parameters']['tau'] = 0.6
            config['parameters']['rho_max'] = 1.0 if condition_type == 'near_vacuum' else 0.95
            
            # Custom extreme initial conditions
            density, velocity = self.create_extreme_initial_conditions(condition_type)
            config['initial_conditions'] = {
                'type': 'custom',
                'density': density.tolist(),
                'velocity': velocity.tolist()
            }
        
        return config
    
    def check_numerical_stability(self, simulation_results):
        """Check for numerical instabilities in simulation results."""
        stability_metrics = {
            'has_nan': False,
            'has_inf': False,
            'density_bounds_violated': False,
            'velocity_bounds_violated': False,
            'mass_conservation_violated': False,
            'max_density_variation': 0.0,
            'max_velocity_variation': 0.0,
            'stability_score': 1.0
        }
        
        if not simulation_results:
            stability_metrics['stability_score'] = 0.0
            return stability_metrics
        
        penalties = 0
        
        for i, state in enumerate(simulation_results):
            density = np.array(state['density'])
            velocity = np.array(state['velocity'])
            
            # Check for NaN or Inf
            if np.any(np.isnan(density)) or np.any(np.isnan(velocity)):
                stability_metrics['has_nan'] = True
                penalties += 0.4
            
            if np.any(np.isinf(density)) or np.any(np.isinf(velocity)):
                stability_metrics['has_inf'] = True  
                penalties += 0.4
            
            # Check physical bounds
            if np.any(density < 0) or np.any(density > 1.0):
                stability_metrics['density_bounds_violated'] = True
                penalties += 0.2
            
            if np.any(velocity < 0) or np.any(velocity > 1.2):
                stability_metrics['velocity_bounds_violated'] = True
                penalties += 0.2
            
            # Track maximum variations
            if i > 0:
                prev_state = simulation_results[i-1]
                prev_density = np.array(prev_state['density'])
                prev_velocity = np.array(prev_state['velocity'])
                
                density_change = np.max(np.abs(density - prev_density))
                velocity_change = np.max(np.abs(velocity - prev_velocity))
                
                stability_metrics['max_density_variation'] = max(
                    stability_metrics['max_density_variation'], density_change
                )
                stability_metrics['max_velocity_variation'] = max(
                    stability_metrics['max_velocity_variation'], velocity_change
                )
                
                # Penalty for excessive variations
                if density_change > 0.3:
                    penalties += 0.1
                if velocity_change > 0.5:
                    penalties += 0.1
        
        # Check mass conservation (simplified)
        if len(simulation_results) >= 2:
            initial_density = np.array(simulation_results[0]['density'])
            final_density = np.array(simulation_results[-1]['density'])
            
            initial_mass = np.sum(initial_density) * (10.0 / len(initial_density))  # dx approximation
            final_mass = np.sum(final_density) * (10.0 / len(final_density))
            
            mass_error = abs(final_mass - initial_mass) / initial_mass if initial_mass > 0 else 0
            if mass_error > 1e-3:  # Lenient for extreme conditions
                stability_metrics['mass_conservation_violated'] = True
                penalties += 0.3
        
        # Calculate overall stability score
        stability_metrics['stability_score'] = max(0.0, 1.0 - penalties)
        
        return stability_metrics
    
    def evaluate_robustness_performance(self, simulation_results, condition_type):
        """Evaluate performance under extreme conditions."""
        performance_metrics = {
            'completion_ratio': 0.0,
            'avg_flow_efficiency': 0.0,
            'solution_quality': 0.0,
            'robustness_score': 0.0
        }
        
        if not simulation_results:
            return performance_metrics
        
        n_completed = len(simulation_results)
        n_expected = 30  # Expected number of time steps
        performance_metrics['completion_ratio'] = min(1.0, n_completed / n_expected)
        
        # Calculate flow efficiency over time
        flow_efficiencies = []
        for state in simulation_results:
            density = np.array(state['density'])
            velocity = np.array(state['velocity'])
            flow = density * velocity
            
            # Theoretical maximum flow (around rho=0.33 for typical AR model)
            theoretical_max_flow = 0.25
            avg_flow = np.mean(flow)
            efficiency = avg_flow / theoretical_max_flow
            flow_efficiencies.append(efficiency)
        
        if flow_efficiencies:
            performance_metrics['avg_flow_efficiency'] = float(np.mean(flow_efficiencies))
        
        # Solution quality based on smoothness and physical consistency
        if len(simulation_results) > 1:
            density_variations = []
            velocity_variations = []
            
            for i in range(1, len(simulation_results)):
                curr_density = np.array(simulation_results[i]['density'])
                prev_density = np.array(simulation_results[i-1]['density'])
                curr_velocity = np.array(simulation_results[i]['velocity'])
                prev_velocity = np.array(simulation_results[i-1]['velocity'])
                
                # Measure smoothness (low variation is good for stability)
                density_var = np.std(curr_density - prev_density)
                velocity_var = np.std(curr_velocity - prev_velocity)
                
                density_variations.append(density_var)
                velocity_variations.append(velocity_var)
            
            # Quality score based on bounded variations
            max_acceptable_density_var = 0.1
            max_acceptable_velocity_var = 0.2
            
            avg_density_var = np.mean(density_variations)
            avg_velocity_var = np.mean(velocity_variations)
            
            density_quality = max(0, 1 - avg_density_var / max_acceptable_density_var)
            velocity_quality = max(0, 1 - avg_velocity_var / max_acceptable_velocity_var)
            
            performance_metrics['solution_quality'] = float((density_quality + velocity_quality) / 2)
        
        # Overall robustness score
        weights = [0.3, 0.3, 0.4]  # completion, efficiency, quality
        scores = [
            performance_metrics['completion_ratio'],
            performance_metrics['avg_flow_efficiency'],
            performance_metrics['solution_quality']
        ]
        
        performance_metrics['robustness_score'] = sum(w * s for w, s in zip(weights, scores))
        
        return performance_metrics
    
    def run_robustness_test(self, condition_type):
        """Run single robustness test for given condition."""
        print(f"\nTesting condition: {condition_type}")
        
        try:
            # Create test configuration
            config = self.create_robustness_config(condition_type)
            
            # Run simulation
            simulation_results = run_mock_simulation(config)
            
            if simulation_results is None:
                print(f"  FAILED: Simulation crashed for {condition_type}")
                return {
                    'success': False,
                    'error': 'Simulation failed',
                    'stability': {'stability_score': 0.0},
                    'performance': {'robustness_score': 0.0}
                }
            
            # Check numerical stability
            stability_metrics = self.check_numerical_stability(simulation_results)
            
            # Evaluate performance
            performance_metrics = self.evaluate_robustness_performance(
                simulation_results, condition_type
            )
            
            # Determine success criteria for robustness tests
            # More lenient than normal validation due to extreme conditions
            success_criteria = [
                stability_metrics['stability_score'] > 0.6,  # Basic stability
                not stability_metrics['has_nan'],
                not stability_metrics['has_inf'],
                performance_metrics['completion_ratio'] > 0.8,  # Complete most of simulation
                performance_metrics['robustness_score'] > 0.4   # Reasonable performance
            ]
            
            test_success = sum(success_criteria) >= 4  # At least 4/5 criteria
            
            results = {
                'success': test_success,
                'stability': stability_metrics,
                'performance': performance_metrics,
                'criteria_met': sum(success_criteria),
                'total_criteria': len(success_criteria),
                'simulation_length': len(simulation_results)
            }
            
            print(f"  Stability score: {stability_metrics['stability_score']:.2f}")
            print(f"  Robustness score: {performance_metrics['robustness_score']:.2f}")
            print(f"  Completion: {performance_metrics['completion_ratio']:.1%}")
            print(f"  Result: {'PASSED' if test_success else 'FAILED'}")
            
            return results
            
        except Exception as e:
            print(f"  ERROR: Exception in {condition_type}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'stability': {'stability_score': 0.0},
                'performance': {'robustness_score': 0.0}
            }
    
    def run_test(self) -> bool:
        """Run robustness validation test."""
        print("=== Section 7.7: Robustness Validation ===")
        print("Testing system robustness under extreme conditions...")
        
        all_results = {}
        overall_success = True
        
        # Test all extreme conditions
        conditions = list(self.extreme_conditions.keys())
        
        for condition in conditions:
            condition_results = self.run_robustness_test(condition)
            all_results[condition] = condition_results
            
            if not condition_results.get('success', False):
                overall_success = False
        
        # Calculate summary metrics
        successful_conditions = sum(1 for r in all_results.values() if r.get('success', False))
        success_rate = (successful_conditions / len(conditions)) * 100
        
        # Average metrics across all conditions (including failed ones for realistic assessment)
        avg_stability_score = []
        avg_robustness_score = []
        avg_completion_ratio = []
        
        for condition, results in all_results.items():
            if 'stability' in results and 'performance' in results:
                avg_stability_score.append(results['stability']['stability_score'])
                avg_robustness_score.append(results['performance']['robustness_score'])
                avg_completion_ratio.append(results['performance'].get('completion_ratio', 0.0))
        
        summary_metrics = {
            'success_rate': success_rate,
            'conditions_passed': successful_conditions,
            'total_conditions': len(conditions),
            'avg_stability_score': np.mean(avg_stability_score) if avg_stability_score else 0.0,
            'avg_robustness_score': np.mean(avg_robustness_score) if avg_robustness_score else 0.0,
            'avg_completion_ratio': np.mean(avg_completion_ratio) if avg_completion_ratio else 0.0
        }
        
        # Store results for LaTeX generation
        self.results = {
            'summary': summary_metrics,
            'conditions': all_results,
            'validation_type': 'robustness',
            'revendications': ['R6']
        }
        
        # Generate LaTeX content
        self.generate_latex_content()
        
        # Robustness validation criteria (lenient due to extreme nature)
        validation_success = (
            success_rate >= 60.0 and  # At least 60% conditions pass (3/5)
            summary_metrics['avg_stability_score'] > 0.6 and
            summary_metrics['avg_robustness_score'] > 0.4 and
            summary_metrics['avg_completion_ratio'] > 0.75
        )
        
        print(f"\n=== Robustness Validation Summary ===")
        print(f"Conditions passed: {successful_conditions}/{len(conditions)} ({success_rate:.1f}%)")
        print(f"Average stability score: {summary_metrics['avg_stability_score']:.2f}")
        print(f"Average robustness score: {summary_metrics['avg_robustness_score']:.2f}")
        print(f"Average completion ratio: {summary_metrics['avg_completion_ratio']:.1%}")
        print(f"Overall validation: {'PASSED' if validation_success else 'FAILED'}")
        
        return validation_success
    
    def generate_latex_content(self):
        """Generate LaTeX content for robustness validation."""
        if not hasattr(self, 'results'):
            return
        
        summary = self.results['summary']
        
        # Main results table
        latex_content = generate_latex_table(
            caption="R\\'esultats Validation Robustesse (Section 7.7)",
            headers=["M\\'etrique", "Valeur", "Seuil", "Statut"],
            rows=[
                ["Taux de succ\\`es conditions", f"{summary['success_rate']:.1f}\\%", 
                 "$\\geq 60\\%$", "PASS" if summary['success_rate'] >= 60.0 else "FAIL"],
                ["Score stabilit\\'e moyen", f"{summary['avg_stability_score']:.2f}", 
                 "$> 0.6$", "PASS" if summary['avg_stability_score'] > 0.6 else "FAIL"],
                ["Score robustesse moyen", f"{summary['avg_robustness_score']:.2f}", 
                 "$> 0.4$", "PASS" if summary['avg_robustness_score'] > 0.4 else "FAIL"],
                ["Ratio compl\\'etion moyen", f"{summary['avg_completion_ratio']:.1%}", 
                 "$> 75\\%$", "PASS" if summary['avg_completion_ratio'] > 0.75 else "FAIL"]
            ]
        )
        
        # Save LaTeX content
        save_validation_results(
            section="7.7",
            content=latex_content,
            results=self.results
        )


def main():
    """Main function to run robustness validation."""
    test = RobustnessValidationTest()
    success = test.run_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()