#!/usr/bin/env python3
"""
Validation Script: Section 7.5 - Digital Twin Validation

Tests for Revendication R4: Reproduction des comportements de trafic observes
Tests for Revendication R6: Robustesse sous conditions degradees

This script validates the digital twin capabilities by:
- Testing behavioral reproduction (traffic patterns, congestion, free flow)
- Testing robustness under degraded conditions (perturbations)
- Cross-scenario validation for consistency

Architecture: Inherits ValidationSection for standardized output structure
"""

import sys
import os
import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from validation_ch7.scripts.validation_utils import (
    ValidationSection, run_real_simulation, setup_publication_style
)
from arz_model.analysis.metrics import (
    compute_mape, compute_rmse, compute_geh, compute_theil_u,
    calculate_total_mass
)
from arz_model.core.physics import calculate_pressure


class DigitalTwinValidationTest(ValidationSection):
    """Digital Twin validation test implementation - inherits ValidationSection."""
    
    def __init__(self, output_base: str = "validation_output/results/local_test"):
        super().__init__(section_name="section_7_5_digital_twin", output_base=output_base)
        
        # Define behavioral patterns for validation (R4)
        self.behavioral_patterns = {
            'free_flow': {
                'description': 'Trafic fluide sans congestion',
                'density_range_m': (10.0e-3, 20.0e-3),  # 10-20 veh/km in SI (veh/m)
                'velocity_range_m': (20.0, 28.0),   # m/s (72-100 km/h)
                'expected_mape_threshold': 25.0
            },
            'congestion': {
                'description': 'Congestion modérée avec onde de choc',
                'density_range_m': (50.0e-3, 80.0e-3),  # 50-80 veh/km
                'velocity_range_m': (8.0, 15.0),    # m/s (29-54 km/h)
                'expected_mape_threshold': 30.0
            },
            'jam_formation': {
                'description': 'Formation et dissolution de bouchon',
                'density_range_m': (80.0e-3, 100.0e-3),  # 80-100 veh/km
                'velocity_range_m': (2.0, 8.0),     # m/s (7-29 km/h)
                'expected_mape_threshold': 40.0
            }
        }
        
        # Robustness test configurations (R6)
        self.perturbation_tests = {
            'density_increase': {
                'description': 'Augmentation densité +50%',
                'multiplier': 1.5,
                'max_convergence_time': 150.0  # seconds
            },
            'velocity_decrease': {
                'description': 'Diminution vitesse -30%',
                'multiplier': 0.7,
                'max_convergence_time': 180.0
            },
            'road_degradation': {
                'description': 'Dégradation qualité route R=1',
                'road_quality': 1,
                'max_convergence_time': 200.0
            }
        }
        
        # Storage for test results
        self.test_results = {}
        
    def create_scenario_config(self, scenario_type: str, grid_size: int = 200, 
                              final_time: float = 100.0, perturbation: dict = None) -> Path:
        """
        Create YAML scenario configuration for real simulation.
        
        Args:
            scenario_type: Type of scenario ('free_flow', 'congestion', 'jam_formation')
            grid_size: Number of grid points
            final_time: Simulation duration in seconds
            perturbation: Optional perturbation configuration for robustness tests
            
        Returns:
            Path to created scenario YAML file
        """
        # Base configuration
        scenario_config = {
            'scenario_name': f'digital_twin_{scenario_type}',
            'N': grid_size,
            'xmin': 0.0,
            'xmax': 5000.0,  # 5 km domain
            't_final': final_time,
            'output_dt': 10.0,
            'CFL': 0.3,  # More conservative
            'boundary_conditions': {
                'left': {'type': 'periodic'},
                'right': {'type': 'periodic'}
            },
            'road': {
                'type': 'uniform',
                'R_val': perturbation.get('road_quality', 2) if perturbation else 2
            }
        }
        
        # Configure initial conditions based on scenario type
        if scenario_type == 'free_flow':
            # Light traffic: 12-15 veh/km in SI units (veh/m)
            rho_m_bg = 12.0e-3
            rho_c_bg = 8.0e-3
            
            if perturbation and 'multiplier' in perturbation:
                rho_m_bg *= perturbation['multiplier']
                rho_c_bg *= perturbation['multiplier']
            
            scenario_config['initial_conditions'] = {
                'type': 'sine_wave_perturbation',
                'R_val': perturbation.get('road_quality', 2) if perturbation else 2,
                'background_state': {
                    'rho_m': rho_m_bg,
                    'rho_c': rho_c_bg
                },
                'perturbation': {
                    'amplitude': 2.0e-3,  # Small fluctuation
                    'wave_number': 1
                }
            }
            scenario_config['parameters'] = {
                'V0_m': 27.8,
                'V0_c': 30.6,
                'tau_m': 0.5,
                'tau_c': 0.6,
                'rho_max_m': 0.15,
                'rho_max_c': 0.12
            }
        
        elif scenario_type == 'congestion':
            # Moderate congestion: 30-70 veh/km
            rho_m_bg = 30.0e-3
            rho_c_bg = 20.0e-3
            rho_m_peak = 70.0e-3
            rho_c_peak = 50.0e-3
            
            if perturbation and 'multiplier' in perturbation:
                mult = perturbation['multiplier']
                rho_m_bg *= mult
                rho_c_bg *= mult
                rho_m_peak *= mult
                rho_c_peak *= mult
            
            # Calculate equilibrium w values for background state
            # For simplicity, use approximate equilibrium: w ≈ V_e
            V0_m = 22.2  # m/s
            V0_c = 25.0  # m/s
            w_m_bg = V0_m  # Approximate
            w_c_bg = V0_c  # Approximate
            
            scenario_config['initial_conditions'] = {
                'type': 'density_hump',  # CORRECTED from 'gaussian_density_pulse'
                'background_state': [rho_m_bg, w_m_bg, rho_c_bg, w_c_bg],  # 4-element list
                'center': 2500.0,
                'width': 500.0,
                'rho_m_max': rho_m_peak,  # Peak density, not amplitude
                'rho_c_max': rho_c_peak
            }
            scenario_config['parameters'] = {
                'V0_m': V0_m,
                'V0_c': V0_c,
                'tau_m': 1.0,
                'tau_c': 1.2,
                'rho_max_m': 0.12,
                'rho_max_c': 0.10
            }
        
        elif scenario_type == 'jam_formation':
            # Heavy jam: 40-90 veh/km
            rho_m_left = 40.0e-3
            rho_c_left = 25.0e-3
            rho_m_right = 90.0e-3
            rho_c_right = 60.0e-3
            
            if perturbation and 'multiplier' in perturbation:
                mult = perturbation['multiplier']
                rho_m_left *= mult
                rho_c_left *= mult
                rho_m_right *= mult
                rho_c_right *= mult
            
            # Calculate equilibrium w values
            V0_m = 19.4  # m/s
            V0_c = 22.2  # m/s
            w_m_left = V0_m * 0.5  # Reduced speed in jam (approximate)
            w_c_left = V0_c * 0.5
            w_m_right = V0_m * 0.2  # Very slow in heavy jam
            w_c_right = V0_c * 0.2
            
            scenario_config['initial_conditions'] = {
                'type': 'riemann',  # CORRECTED from 'step_density'
                'U_L': [rho_m_left, w_m_left, rho_c_left, w_c_left],  # Left state
                'U_R': [rho_m_right, w_m_right, rho_c_right, w_c_right],  # Right state
                'split_pos': 2500.0  # Discontinuity position
            }
            scenario_config['parameters'] = {
                'V0_m': V0_m,
                'V0_c': V0_c,
                'tau_m': 1.5,
                'tau_c': 1.8,
                'rho_max_m': 0.15,
                'rho_max_c': 0.12
            }
        
        # Save scenario configuration
        suffix = f"_{perturbation.get('name', 'nominal')}" if perturbation else "_nominal"
        scenario_path = self.scenarios_dir / f"{scenario_type}{suffix}.yml"
        with open(scenario_path, 'w') as f:
            yaml.dump(scenario_config, f, default_flow_style=False)
        
        print(f"[SCENARIO] Created: {scenario_path.name}")
        return scenario_path
    
    def test_behavioral_reproduction(self) -> dict:
        """
        Test R4: Behavioral reproduction across different traffic scenarios.
        
        Returns:
            dict: Test results with metrics for each scenario
        """
        print("\n=== Test R4: Behavioral Reproduction ===")
        
        results = {}
        overall_success = True
        
        for scenario_name, pattern in self.behavioral_patterns.items():
            print(f"\n[SCENARIO] {scenario_name}: {pattern['description']}")
            
            try:
                # Create scenario configuration
                scenario_path = self.create_scenario_config(
                    scenario_name, 
                    grid_size=200, 
                    final_time=100.0
                )
                
                # Run real simulation - use fallback path resolution in run_real_simulation()
                sim_result = run_real_simulation(
                    str(scenario_path),
                    base_config_path=None,  # Let run_real_simulation() find the correct path
                    device='cpu'
                )
                
                if sim_result is None:
                    print(f"  [FAILED] Simulation failed for {scenario_name}")
                    results[scenario_name] = {'success': False, 'error': 'Simulation failed'}
                    overall_success = False
                    continue
                
                # Extract simulation data
                times = sim_result['times']
                states = sim_result['states']
                grid = sim_result['grid']
                params = sim_result['params']
                
                # Analyze final state
                final_state = states[-1]
                rho_m_final = final_state[0, :]  # Motorcycle density
                w_m_final = final_state[1, :]    # Motorcycle Lagrangian variable
                rho_c_final = final_state[2, :]  # Car density
                
                # Calculate pressure to extract velocity: v = w - p
                p_m_final, _ = calculate_pressure(
                    rho_m_final, rho_c_final,
                    params.alpha, params.rho_jam, params.epsilon,
                    params.K_m, params.gamma_m,
                    params.K_c, params.gamma_c
                )
                
                # Extract velocity using correct ARZ formula: v = w - p
                v_m_final = w_m_final - p_m_final
                
                # Calculate metrics
                avg_rho_m = np.mean(rho_m_final)
                avg_v_m = np.mean(v_m_final[rho_m_final > 0.001])  # Only where traffic exists
                std_rho_m = np.std(rho_m_final)
                std_v_m = np.std(v_m_final[rho_m_final > 0.001])
                
                # Check if behavior matches expected pattern
                rho_in_range = (pattern['density_range_m'][0] <= avg_rho_m <= 
                               pattern['density_range_m'][1])
                v_in_range = (pattern['velocity_range_m'][0] <= avg_v_m <= 
                             pattern['velocity_range_m'][1])
                
                # Mass conservation check
                initial_mass = sim_result['mass_conservation']['initial_mass_m']
                final_mass = sim_result['mass_conservation']['final_mass_m']
                mass_error = abs(final_mass - initial_mass) / initial_mass * 100
                
                # Store results
                results[scenario_name] = {
                    'success': rho_in_range and v_in_range and mass_error < 1.0,
                    'avg_density': float(avg_rho_m),
                    'avg_velocity': float(avg_v_m),
                    'std_density': float(std_rho_m),
                    'std_velocity': float(std_v_m),
                    'density_in_range': rho_in_range,
                    'velocity_in_range': v_in_range,
                    'mass_conservation_error': float(mass_error),
                    'simulation_time': float(times[-1])
                }
                
                status = "PASSED" if results[scenario_name]['success'] else "FAILED"
                print(f"  [{status}] ρ_m={avg_rho_m:.4f} veh/m, v_m={avg_v_m:.2f} m/s, mass_error={mass_error:.3f}%")
                
                if not results[scenario_name]['success']:
                    overall_success = False
                    
            except Exception as e:
                print(f"  [ERROR] Exception: {str(e)}")
                results[scenario_name] = {'success': False, 'error': str(e)}
                overall_success = False
        
        # Calculate summary metrics
        successful_scenarios = sum(1 for r in results.values() if r.get('success', False))
        
        summary = {
            'test_name': 'behavioral_reproduction',
            'revendication': 'R4',
            'scenarios_tested': len(self.behavioral_patterns),
            'scenarios_passed': successful_scenarios,
            'success_rate': (successful_scenarios / len(self.behavioral_patterns)) * 100,
            'overall_success': overall_success,
            'scenario_results': results
        }
        
        self.test_results['behavioral_reproduction'] = summary
        return summary
    
    def test_robustness_degraded_conditions(self) -> dict:
        """
        Test R6: Robustness under degraded conditions (perturbations).
        
        Returns:
            dict: Test results with convergence times and stability metrics
        """
        print("\n=== Test R6: Robustness Under Degraded Conditions ===")
        
        results = {}
        overall_success = True
        
        # Test free_flow scenario with different perturbations
        base_scenario = 'free_flow'
        
        for pert_name, pert_config in self.perturbation_tests.items():
            print(f"\n[PERTURBATION] {pert_name}: {pert_config['description']}")
            
            try:
                # Create perturbation parameters
                perturbation = {'name': pert_name}
                
                if pert_name == 'density_increase':
                    perturbation['multiplier'] = pert_config['multiplier']
                elif pert_name == 'velocity_decrease':
                    perturbation['velocity_multiplier'] = pert_config['multiplier']
                elif pert_name == 'road_degradation':
                    perturbation['road_quality'] = pert_config['road_quality']
                
                # Create perturbed scenario
                scenario_path = self.create_scenario_config(
                    base_scenario,
                    grid_size=200,
                    final_time=200.0,  # Longer time for convergence
                    perturbation=perturbation
                )
                
                # Run simulation - use fallback path resolution
                sim_result = run_real_simulation(
                    str(scenario_path),
                    base_config_path=None,  # Let run_real_simulation() find the correct path
                    device='cpu'
                )
                
                if sim_result is None:
                    print(f"  [FAILED] Simulation failed")
                    results[pert_name] = {'success': False, 'error': 'Simulation failed'}
                    overall_success = False
                    continue
                
                # Analyze convergence
                times = sim_result['times']
                states = sim_result['states']
                
                # Check for numerical stability (no NaN or explosion)
                has_nan = any(np.isnan(state).any() for state in states)
                has_explosion = any(np.max(np.abs(state)) > 1e3 for state in states)
                
                numerical_stable = not (has_nan or has_explosion)
                
                # Estimate convergence time (when std deviation stabilizes)
                convergence_time = times[-1]  # Default to final time
                for i in range(10, len(states)):
                    rho_std = np.std(states[i][0, :])
                    if i > 20 and abs(rho_std - np.std(states[i-10][0, :])) < 0.001:
                        convergence_time = times[i]
                        break
                
                converged = convergence_time < pert_config['max_convergence_time']
                
                # Final RMSE
                final_state = states[-1]
                rho_final = final_state[0, :]
                rmse_final = np.sqrt(np.mean((rho_final - np.mean(rho_final))**2))
                
                results[pert_name] = {
                    'success': numerical_stable and converged,
                    'numerical_stable': numerical_stable,
                    'converged': converged,
                    'convergence_time': float(convergence_time),
                    'max_convergence_time': pert_config['max_convergence_time'],
                    'final_rmse': float(rmse_final),
                    'has_nan': has_nan,
                    'has_explosion': has_explosion
                }
                
                status = "PASSED" if results[pert_name]['success'] else "FAILED"
                print(f"  [{status}] Conv_time={convergence_time:.1f}s (max={pert_config['max_convergence_time']}s), "
                      f"RMSE={rmse_final:.4f}, Stable={numerical_stable}")
                
                if not results[pert_name]['success']:
                    overall_success = False
                    
            except Exception as e:
                print(f"  [ERROR] Exception: {str(e)}")
                results[pert_name] = {'success': False, 'error': str(e)}
                overall_success = False
        
        successful_tests = sum(1 for r in results.values() if r.get('success', False))
        
        summary = {
            'test_name': 'robustness_degraded_conditions',
            'revendication': 'R6',
            'perturbations_tested': len(self.perturbation_tests),
            'perturbations_passed': successful_tests,
            'success_rate': (successful_tests / len(self.perturbation_tests)) * 100,
            'overall_success': overall_success,
            'perturbation_results': results
        }
        
        self.test_results['robustness'] = summary
        return summary
    
    def test_cross_scenario_validation(self) -> dict:
        """
        Cross-scenario consistency validation.
        
        Returns:
            dict: Consistency metrics across scenarios
        """
        print("\n=== Test: Cross-Scenario Validation ===")
        
        # This test checks that the model maintains consistency
        # (e.g., higher density → lower velocity across all scenarios)
        
        if 'behavioral_reproduction' not in self.test_results:
            return {'success': False, 'error': 'Run behavioral_reproduction test first'}
        
        behavioral = self.test_results['behavioral_reproduction']['scenario_results']
        
        # Extract density and velocity relationships
        scenarios_data = []
        for name, result in behavioral.items():
            if result.get('success'):
                scenarios_data.append({
                    'name': name,
                    'density': result['avg_density'],
                    'velocity': result['avg_velocity']
                })
        
        if len(scenarios_data) < 2:
            return {'success': False, 'error': 'Not enough successful scenarios'}
        
        # Check fundamental diagram relationship: higher density → lower velocity
        sorted_by_density = sorted(scenarios_data, key=lambda x: x['density'])
        
        monotonic = True
        for i in range(len(sorted_by_density) - 1):
            if sorted_by_density[i+1]['velocity'] > sorted_by_density[i]['velocity']:
                monotonic = False
                break
        
        results = {
            'success': monotonic,
            'monotonic_relationship': monotonic,
            'scenarios_analyzed': len(scenarios_data),
            'density_range': [sorted_by_density[0]['density'], sorted_by_density[-1]['density']],
            'velocity_range': [sorted_by_density[-1]['velocity'], sorted_by_density[0]['velocity']]
        }
        
        status = "PASSED" if monotonic else "FAILED"
        print(f"  [{status}] Fundamental diagram monotonicity: {monotonic}")
        
        self.test_results['cross_scenario'] = results
        return results
    
    def generate_digital_twin_figures(self):
        """Generate all publication-quality figures for Section 7.5."""
        print("\n=== Generating Figures ===")
        
        setup_publication_style()
        
        # Figure 1: Behavioral patterns comparison
        self._generate_behavioral_patterns_figure()
        
        # Figure 2: Robustness perturbations response
        self._generate_robustness_figure()
        
        # Figure 3: Fundamental diagram
        self._generate_fundamental_diagram_figure()
        
        # Figure 4: Metrics summary
        self._generate_metrics_summary_figure()
        
        print(f"[FIGURES] Generated 4 figures in {self.figures_dir}")
    
    def _generate_behavioral_patterns_figure(self):
        """Figure 1: Behavioral patterns for 3 scenarios."""
        if 'behavioral_reproduction' not in self.test_results:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        behavioral = self.test_results['behavioral_reproduction']['scenario_results']
        
        for idx, (scenario_name, result) in enumerate(behavioral.items()):
            if not result.get('success'):
                continue
            
            ax = axes[idx]
            
            # Simple bar chart showing density and velocity
            metrics = ['Densité\n(veh/m)', 'Vitesse\n(m/s)']
            values = [result['avg_density'] * 100, result['avg_velocity']]  # Scale density for visibility
            colors = ['#1f77b4', '#ff7f0e']
            
            bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_title(f"{scenario_name.replace('_', ' ').title()}", fontweight='bold')
            ax.set_ylabel('Valeur')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        fig.savefig(self.figures_dir / 'fig_behavioral_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] fig_behavioral_patterns.png")
    
    def _generate_robustness_figure(self):
        """Figure 2: Robustness under perturbations."""
        if 'robustness' not in self.test_results:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        robustness = self.test_results['robustness']['perturbation_results']
        
        # Convergence times
        pert_names = []
        conv_times = []
        max_times = []
        colors = []
        
        for name, result in robustness.items():
            if result.get('success') is not None:
                pert_names.append(name.replace('_', '\n'))
                conv_times.append(result.get('convergence_time', 0))
                max_times.append(result.get('max_convergence_time', 0))
                colors.append('green' if result['success'] else 'red')
        
        x = np.arange(len(pert_names))
        width = 0.35
        
        ax1.bar(x - width/2, conv_times, width, label='Temps convergence', color=colors, alpha=0.7)
        ax1.bar(x + width/2, max_times, width, label='Seuil max', color='gray', alpha=0.5)
        ax1.set_xlabel('Perturbation')
        ax1.set_ylabel('Temps (s)')
        ax1.set_title('Temps de Convergence sous Perturbations', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(pert_names, fontsize=8)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # RMSE final
        rmse_values = [result.get('final_rmse', 0) for result in robustness.values() 
                       if result.get('success') is not None]
        
        ax2.bar(pert_names, rmse_values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Perturbation')
        ax2.set_ylabel('RMSE Final')
        ax2.set_title('Stabilité Numérique (RMSE)', fontweight='bold')
        ax2.set_xticklabels(pert_names, fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(self.figures_dir / 'fig_robustness_perturbations.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] fig_robustness_perturbations.png")
    
    def _generate_fundamental_diagram_figure(self):
        """Figure 3: Fundamental diagram from behavioral tests."""
        if 'behavioral_reproduction' not in self.test_results:
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        behavioral = self.test_results['behavioral_reproduction']['scenario_results']
        
        densities = []
        velocities = []
        flows = []
        labels = []
        
        for name, result in behavioral.items():
            if result.get('success'):
                rho = result['avg_density']
                v = result['avg_velocity']
                densities.append(rho)
                velocities.append(v)
                flows.append(rho * v)
                labels.append(name.replace('_', ' ').title())
        
        # Plot fundamental diagram
        colors = ['green', 'orange', 'red']
        for i, (rho, q, label) in enumerate(zip(densities, flows, labels)):
            ax.scatter(rho * 1000, q * 3600, s=200, c=colors[i], 
                      label=label, alpha=0.7, edgecolors='black', linewidths=2)
        
        ax.set_xlabel('Densité (veh/km)', fontsize=12)
        ax.set_ylabel('Débit (veh/h)', fontsize=12)
        ax.set_title('Diagramme Fondamental - Digital Twin', fontweight='bold', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(self.figures_dir / 'fig_fundamental_diagram.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] fig_fundamental_diagram.png")
    
    def _generate_metrics_summary_figure(self):
        """Figure 4: Summary metrics bar chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Collect all success rates
        test_names = []
        success_rates = []
        colors_list = []
        
        if 'behavioral_reproduction' in self.test_results:
            test_names.append('R4: Behavioral\nReproduction')
            success_rates.append(self.test_results['behavioral_reproduction']['success_rate'])
            colors_list.append('green' if self.test_results['behavioral_reproduction']['overall_success'] else 'red')
        
        if 'robustness' in self.test_results:
            test_names.append('R6: Robustness\nDegraded Conditions')
            success_rates.append(self.test_results['robustness']['success_rate'])
            colors_list.append('green' if self.test_results['robustness']['overall_success'] else 'red')
        
        if 'cross_scenario' in self.test_results:
            test_names.append('Cross-Scenario\nValidation')
            success_rates.append(100.0 if self.test_results['cross_scenario']['success'] else 0.0)
            colors_list.append('green' if self.test_results['cross_scenario']['success'] else 'red')
        
        bars = ax.bar(test_names, success_rates, color=colors_list, alpha=0.7, edgecolor='black', linewidth=2)
        ax.axhline(y=70, color='orange', linestyle='--', linewidth=2, label='Seuil Acceptable (70%)')
        ax.set_ylabel('Taux de Succès (%)', fontsize=12)
        ax.set_title('Section 7.5 - Validation Jumeau Numérique: Résumé Métriques', fontweight='bold', fontsize=14)
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        fig.savefig(self.figures_dir / 'fig_digital_twin_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] fig_digital_twin_metrics.png")
    
    def save_digital_twin_data(self):
        """Save CSV metrics and session data."""
        print("\n=== Saving Data ===")
        
        # CSV 1: Behavioral metrics
        if 'behavioral_reproduction' in self.test_results:
            behavioral_data = []
            for scenario, result in self.test_results['behavioral_reproduction']['scenario_results'].items():
                behavioral_data.append({
                    'scenario': scenario,
                    'avg_density_veh_m': result.get('avg_density', 0),
                    'avg_velocity_m_s': result.get('avg_velocity', 0),
                    'std_density': result.get('std_density', 0),
                    'std_velocity': result.get('std_velocity', 0),
                    'mass_conservation_error_pct': result.get('mass_conservation_error', 0),
                    'success': result.get('success', False)
                })
            
            df_behavioral = pd.DataFrame(behavioral_data)
            csv_path = self.metrics_dir / 'behavioral_metrics.csv'
            df_behavioral.to_csv(csv_path, index=False)
            print(f"  [OK] {csv_path.name}")
        
        # CSV 2: Robustness metrics
        if 'robustness' in self.test_results:
            robustness_data = []
            for pert, result in self.test_results['robustness']['perturbation_results'].items():
                robustness_data.append({
                    'perturbation': pert,
                    'convergence_time_s': result.get('convergence_time', 0),
                    'max_convergence_time_s': result.get('max_convergence_time', 0),
                    'final_rmse': result.get('final_rmse', 0),
                    'numerical_stable': result.get('numerical_stable', False),
                    'converged': result.get('converged', False),
                    'success': result.get('success', False)
                })
            
            df_robustness = pd.DataFrame(robustness_data)
            csv_path = self.metrics_dir / 'robustness_metrics.csv'
            df_robustness.to_csv(csv_path, index=False)
            print(f"  [OK] {csv_path.name}")
        
        # CSV 3: Summary metrics
        summary_data = {
            'test_name': [],
            'revendication': [],
            'tests_total': [],
            'tests_passed': [],
            'success_rate_pct': [],
            'overall_success': []
        }
        
        for test_key in ['behavioral_reproduction', 'robustness']:
            if test_key in self.test_results:
                test_data = self.test_results[test_key]
                summary_data['test_name'].append(test_data['test_name'])
                summary_data['revendication'].append(test_data['revendication'])
                summary_data['tests_total'].append(test_data.get('scenarios_tested') or test_data.get('perturbations_tested'))
                summary_data['tests_passed'].append(test_data.get('scenarios_passed') or test_data.get('perturbations_passed'))
                summary_data['success_rate_pct'].append(test_data['success_rate'])
                summary_data['overall_success'].append(test_data['overall_success'])
        
        df_summary = pd.DataFrame(summary_data)
        csv_path = self.metrics_dir / 'summary_metrics.csv'
        df_summary.to_csv(csv_path, index=False)
        print(f"  [OK] {csv_path.name}")
    
    def generate_section_7_5_latex(self):
        """Generate enriched LaTeX content for Section 7.5."""
        print("\n=== Generating LaTeX Content ===")
        
        latex_content = r"""\subsection{Validation Jumeau Numérique (Section 7.5)}

\subsubsection{Objectifs}
Cette section valide les revendications suivantes:
\begin{itemize}
    \item \textbf{R4:} Reproduction des comportements de trafic observés
    \item \textbf{R6:} Robustesse sous conditions dégradées
\end{itemize}

\subsubsection{Méthodologie}

La validation du jumeau numérique comprend trois volets:

\paragraph{Test 1: Reproduction Comportementale (R4)}
Trois scénarios de trafic typiques sont simulés:
\begin{enumerate}
    \item \textbf{Trafic fluide:} Densité faible (10-20 veh/km), vitesses élevées (72-100 km/h)
    \item \textbf{Congestion modérée:} Densité moyenne (50-80 veh/km), vitesses réduites (29-54 km/h)
    \item \textbf{Formation de bouchon:} Densité élevée (80-100 veh/km), vitesses très faibles (7-29 km/h)
\end{enumerate}

Pour chaque scénario, on vérifie que:
\begin{itemize}
    \item La densité moyenne respecte la plage attendue
    \item La vitesse moyenne correspond au régime de trafic
    \item La conservation de la masse est assurée (erreur < 1\%)
\end{itemize}

\paragraph{Test 2: Robustesse sous Perturbations (R6)}
Trois perturbations sont appliquées au scénario de trafic fluide:
\begin{enumerate}
    \item \textbf{Augmentation de densité +50\%:} Test de la réponse à une demande accrue
    \item \textbf{Diminution de vitesse -30\%:} Simulation de conditions météorologiques défavorables
    \item \textbf{Dégradation de route (R=1):} Impact de la qualité de l'infrastructure
\end{enumerate}

Critères de validation:
\begin{itemize}
    \item Stabilité numérique (absence de NaN ou explosions)
    \item Temps de convergence < seuils définis (150-200s)
    \item RMSE final acceptable
\end{itemize}

\paragraph{Test 3: Validation Croisée}
Vérification de la cohérence du diagramme fondamental: la relation densité-vitesse doit être décroissante monotone.

\subsubsection{Résultats}

"""
        
        # Add results tables
        if 'behavioral_reproduction' in self.test_results:
            behavioral = self.test_results['behavioral_reproduction']
            
            latex_content += r"""
\begin{table}[htbp]
\centering
\caption{Résultats R4: Reproduction Comportementale}
\begin{tabular}{lccc}
\toprule
\textbf{Scénario} & \textbf{Densité (veh/m)} & \textbf{Vitesse (m/s)} & \textbf{Statut} \\
\midrule
"""
            
            for scenario, result in behavioral['scenario_results'].items():
                if 'avg_density' in result:
                    status = "PASS" if result['success'] else "FAIL"
                    latex_content += f"{scenario.replace('_', ' ').title()} & {result['avg_density']:.4f} & {result['avg_velocity']:.2f} & {status} \\\\\n"
            
            latex_content += r"""\bottomrule
\end{tabular}
\end{table}

"""
            
            latex_content += f"""
\\textbf{{Taux de succès R4:}} {behavioral['success_rate']:.1f}\\% ({behavioral['scenarios_passed']}/{behavioral['scenarios_tested']} scénarios validés)

"""
        
        if 'robustness' in self.test_results:
            robustness = self.test_results['robustness']
            
            latex_content += r"""
\begin{table}[htbp]
\centering
\caption{Résultats R6: Robustesse sous Perturbations}
\begin{tabular}{lccc}
\toprule
\textbf{Perturbation} & \textbf{Temps Conv. (s)} & \textbf{RMSE Final} & \textbf{Statut} \\
\midrule
"""
            
            for pert, result in robustness['perturbation_results'].items():
                if 'convergence_time' in result:
                    status = "PASS" if result['success'] else "FAIL"
                    latex_content += f"{pert.replace('_', ' ').title()} & {result['convergence_time']:.1f} & {result['final_rmse']:.4f} & {status} \\\\\n"
            
            latex_content += r"""\bottomrule
\end{tabular}
\end{table}

"""
            
            latex_content += f"""
\\textbf{{Taux de succès R6:}} {robustness['success_rate']:.1f}\\% ({robustness['perturbations_passed']}/{robustness['perturbations_tested']} perturbations validées)

"""
        
        # Add figures
        latex_content += r"""
\subsubsection{Visualisations}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{images/fig_behavioral_patterns.png}
\caption{Patterns comportementaux pour les trois scénarios de trafic (densité et vitesse moyennes)}
\label{fig:behavioral_patterns}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{images/fig_robustness_perturbations.png}
\caption{Robustesse sous perturbations: temps de convergence et RMSE final}
\label{fig:robustness_perturbations}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{images/fig_fundamental_diagram.png}
\caption{Diagramme fondamental validant la cohérence physique du modèle}
\label{fig:fundamental_diagram}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{images/fig_digital_twin_metrics.png}
\caption{Résumé des métriques de validation pour la Section 7.5}
\label{fig:digital_twin_metrics}
\end{figure}

\subsubsection{Discussion}

"""
        
        # Add discussion based on results
        overall_behavioral = self.test_results.get('behavioral_reproduction', {}).get('overall_success', False)
        overall_robustness = self.test_results.get('robustness', {}).get('overall_success', False)
        
        if overall_behavioral and overall_robustness:
            latex_content += r"""
\textbf{Forces:}
\begin{itemize}
    \item Reproduction fidèle des patterns comportementaux observés (R4 validé)
    \item Robustesse confirmée sous conditions dégradées (R6 validé)
    \item Cohérence du diagramme fondamental
    \item Stabilité numérique assurée
\end{itemize}
"""
        else:
            latex_content += r"""
\textbf{Observations:}
\begin{itemize}
    \item Certains scénarios nécessitent des ajustements de paramètres
    \item Temps de convergence variables selon les perturbations
    \item Nécessité d'affiner les seuils de validation
\end{itemize}
"""
        
        latex_content += r"""
\textbf{Limitations:}
\begin{itemize}
    \item Tests basés sur simulations (pas de données réelles pour R4)
    \item Gamme de perturbations limitée pour R6
    \item Validation sur domaine 1D uniquement
\end{itemize}

\textbf{Améliorations possibles:}
\begin{itemize}
    \item Intégration de données de capteurs réels pour R4
    \item Extension des tests de robustesse (conditions météo extrêmes)
    \item Validation sur réseaux complexes 2D
    \item Calibration spécifique par type de route
\end{itemize}

\subsubsection{Conclusion}

"""
        
        if overall_behavioral and overall_robustness:
            latex_content += r"""La Section 7.5 valide avec succès les revendications R4 (reproduction comportementale) et R6 (robustesse). Le jumeau numérique démontre sa capacité à reproduire fidèlement les patterns de trafic observés et maintient sa stabilité sous conditions dégradées. Ces résultats confirment la fiabilité du modèle ARZ pour des applications de simulation prédictive en temps réel.
"""
        else:
            latex_content += r"""La Section 7.5 fournit une validation partielle des revendications R4 et R6. Des ajustements supplémentaires sont nécessaires pour améliorer la robustesse globale du jumeau numérique.
"""
        
        # Save LaTeX content
        latex_path = self.latex_dir / 'section_7_5_digital_twin_content.tex'
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        print(f"  [OK] {latex_path.name}")
        
        # Also save to chapters/partie3/ for thesis integration
        thesis_latex_dir = project_root / "chapters" / "partie3"
        thesis_latex_dir.mkdir(parents=True, exist_ok=True)
        thesis_latex_path = thesis_latex_dir / 'section_7_5_digital_twin_content.tex'
        shutil.copy(latex_path, thesis_latex_path)
        print(f"  [OK] Copied to {thesis_latex_path}")
    
    def copy_figures_to_thesis(self):
        """Copy figures to chapters/partie3/images/ for thesis integration."""
        print("\n=== Copying Figures to Thesis ===")
        
        thesis_images_dir = project_root / "chapters" / "partie3" / "images"
        thesis_images_dir.mkdir(parents=True, exist_ok=True)
        
        for fig_file in self.figures_dir.glob('fig_*.png'):
            dest = thesis_images_dir / fig_file.name
            shutil.copy(fig_file, dest)
            print(f"  [OK] {fig_file.name} -> {dest}")
    
    def run_all_tests(self) -> bool:
        """Run all validation tests and generate outputs."""
        print("\n" + "="*80)
        print("SECTION 7.5 - DIGITAL TWIN VALIDATION")
        print("="*80)
        
        # Run tests
        test1_result = self.test_behavioral_reproduction()
        test2_result = self.test_robustness_degraded_conditions()
        test3_result = self.test_cross_scenario_validation()
        
        # Generate outputs
        self.generate_digital_twin_figures()
        self.save_digital_twin_data()
        self.generate_section_7_5_latex()
        
        # Save session summary
        additional_info = {
            'test_status': {
                'behavioral_reproduction': test1_result.get('overall_success', False),
                'robustness': test2_result.get('overall_success', False),
                'cross_scenario': test3_result.get('success', False)
            },
            'overall_validation': (
                test1_result.get('overall_success', False) and 
                test2_result.get('overall_success', False) and
                test3_result.get('success', False)
            )
        }
        self.save_session_summary(additional_info)
        
        # Copy figures to thesis
        self.copy_figures_to_thesis()
        
        # Final status
        overall_success = additional_info['overall_validation']
        
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        print(f"R4 Behavioral Reproduction: {'PASSED' if test1_result['overall_success'] else 'FAILED'}")
        print(f"R6 Robustness Degraded:     {'PASSED' if test2_result['overall_success'] else 'FAILED'}")
        print(f"Cross-Scenario Validation:   {'PASSED' if test3_result['success'] else 'FAILED'}")
        print(f"\nOVERALL STATUS: {'PASSED V' if overall_success else 'FAILED X'}")
        print("="*80)
        
        return overall_success


def main():
    """Main function to run digital twin validation."""
    test = DigitalTwinValidationTest()
    success = test.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
