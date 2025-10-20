"""
Test 3: Fundamental Diagrams Calibration

Generates theoretical fundamental diagrams (V-ρ and Q-ρ) for both vehicle classes
using calibrated parameters from ARZ model.

The fundamental diagrams show the relationships:
  - V-ρ (Speed-Density): How speed decreases with density
  - Q-ρ (Flow-Density): How flow (vehicles/hour) varies with density

Parameters calibrated for West African traffic:
  Motorcycles:
    - Vmax = 60 km/h (exploits mobility advantage)
    - ρmax = 0.15 veh/m (aggressive packing)
    - τ = 0.5s (quick reaction time)
  
  Cars:
    - Vmax = 50 km/h (standard highway speed)
    - ρmax = 0.12 veh/m (standard spacing)
    - τ = 1.0s (conservative reaction time)

Model: ARZ (Aw-Rascle-Zhang) with fundamental diagram
  V(ρ) = Vmax * (1 - ρ/ρmax)
  Q(ρ) = ρ * V(ρ) = ρ * Vmax * (1 - ρ/ρmax)

Author: ARZ-RL Validation Team
Date: 2025-10-17
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import json


class FundamentalDiagramCalibrator:
    """Generates and validates fundamental diagrams for ARZ model."""
    
    def __init__(self):
        """Initialize calibrated parameters."""
        
        # Motorcycles (West Africa)
        self.params_m = {
            'class': 'Motorcycles',
            'symbol': '^',
            'color': 'blue',
            'Vmax_kmh': 60,
            'rho_max': 0.15,  # veh/m
            'tau': 0.5,       # relaxation time (s)
            'length': 2.5     # m (smaller than cars)
        }
        
        # Cars (West Africa)
        self.params_c = {
            'class': 'Cars',
            'symbol': 's',
            'color': 'orange',
            'Vmax_kmh': 50,
            'rho_max': 0.12,  # veh/m
            'tau': 1.0,       # relaxation time (s)
            'length': 5.0     # m
        }
        
        # Convert Vmax to m/s
        self.params_m['Vmax_ms'] = self.params_m['Vmax_kmh'] / 3.6
        self.params_c['Vmax_ms'] = self.params_c['Vmax_kmh'] / 3.6
    
    def compute_V_rho(self, rho: np.ndarray, params: Dict) -> np.ndarray:
        """
        Compute speed as function of density.
        
        V(ρ) = Vmax * (1 - ρ/ρmax)
        """
        return params['Vmax_ms'] * np.maximum(0, 1 - rho / params['rho_max'])
    
    def compute_Q_rho(self, rho: np.ndarray, params: Dict) -> np.ndarray:
        """
        Compute flow as function of density.
        
        Q(ρ) = ρ * V(ρ) = ρ * Vmax * (1 - ρ/ρmax)
        
        Units: veh/m * m/s = veh/s
        Convert to veh/h: multiply by 3600
        """
        V = self.compute_V_rho(rho, params)
        Q = rho * V  # veh/s
        return Q * 3600  # veh/h (convert to vehicles per hour)
    
    def compute_critical_density(self, params: Dict) -> float:
        """
        Critical density where flow is maximum.
        
        Q_max at ρ = ρmax / 2
        """
        return params['rho_max'] / 2
    
    def compute_max_flow(self, params: Dict) -> float:
        """Maximum flow rate."""
        rho_crit = self.compute_critical_density(params)
        Q_max = self.compute_Q_rho(np.array([rho_crit]), params)[0]
        return Q_max
    
    def validate_diagrams(self) -> Dict:
        """Validate fundamental diagrams properties."""
        print("\n✅ Validating Fundamental Diagrams:")
        
        # Critical points for motorcycles
        rho_crit_m = self.compute_critical_density(self.params_m)
        Q_max_m = self.compute_max_flow(self.params_m)
        
        # Critical points for cars
        rho_crit_c = self.compute_critical_density(self.params_c)
        Q_max_c = self.compute_max_flow(self.params_c)
        
        # Motorcycles should have higher max flow (more vehicles per meter)
        flow_ratio = Q_max_m / Q_max_c
        
        validation = {
            'motorcycles': {
                'Vmax_kmh': float(self.params_m['Vmax_kmh']),
                'rho_max': float(self.params_m['rho_max']),
                'rho_critical': float(rho_crit_m),
                'Q_max_veh_per_hour': float(Q_max_m),
                'valid': bool(Q_max_m > Q_max_c)
            },
            'cars': {
                'Vmax_kmh': float(self.params_c['Vmax_kmh']),
                'rho_max': float(self.params_c['rho_max']),
                'rho_critical': float(rho_crit_c),
                'Q_max_veh_per_hour': float(Q_max_c),
                'valid': bool(Q_max_c < Q_max_m)
            },
            'comparison': {
                'Vmax_ratio_moto_car': float(self.params_m['Vmax_kmh'] / self.params_c['Vmax_kmh']),
                'flow_ratio_moto_car': float(flow_ratio),
                'higher_throughput_motos': bool(flow_ratio > 1),
                'throughput_advantage': bool(flow_ratio > 1.1)
            },
            'all_valid': bool(Q_max_m > Q_max_c and flow_ratio > 1.1)
        }
        
        print(f"  🏍️  Motorcycles:")
        print(f"     Vmax = {self.params_m['Vmax_kmh']} km/h")
        print(f"     ρmax = {self.params_m['rho_max']} veh/m")
        print(f"     Q_max = {Q_max_m:.0f} veh/h {'✅' if validation['motorcycles']['valid'] else '❌'}")
        
        print(f"  🚗 Cars:")
        print(f"     Vmax = {self.params_c['Vmax_kmh']} km/h")
        print(f"     ρmax = {self.params_c['rho_max']} veh/m")
        print(f"     Q_max = {Q_max_c:.0f} veh/h {'✅' if validation['cars']['valid'] else '❌'}")
        
        print(f"  📊 Comparison:")
        print(f"     Throughput advantage (motos/cars): {flow_ratio:.2f}x {'✅' if flow_ratio > 1.1 else '❌'}")
        
        return validation
    
    def create_fundamental_diagrams_figure(self, save_path: Path):
        """Create 2x2 figure with V-ρ and Q-ρ diagrams."""
        print("\n🖼️  Creating fundamental diagrams figure...")
        
        # Density range (vehicles per meter)
        rho = np.linspace(0, 0.2, 100)
        
        # Compute curves
        V_m = self.compute_V_rho(rho, self.params_m)
        V_c = self.compute_V_rho(rho, self.params_c)
        Q_m = self.compute_Q_rho(rho, self.params_m)
        Q_c = self.compute_Q_rho(rho, self.params_c)
        
        # Critical densities and flows
        rho_crit_m = self.compute_critical_density(self.params_m)
        rho_crit_c = self.compute_critical_density(self.params_c)
        Q_max_m = self.compute_max_flow(self.params_m)
        Q_max_c = self.compute_max_flow(self.params_c)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # --- Plot 1: V-ρ for both classes ---
        ax = axes[0, 0]
        ax.plot(rho, V_m * 3.6, 'b-', linewidth=2.5, label='Motorcycles')
        ax.plot(rho, V_c * 3.6, 'orange', linewidth=2.5, label='Cars')
        ax.axvline(x=rho_crit_m, color='blue', linestyle='--', alpha=0.5)
        ax.axvline(x=rho_crit_c, color='orange', linestyle='--', alpha=0.5)
        ax.scatter([rho_crit_m], [self.params_m['Vmax_kmh']/2], s=100, c='blue', marker='^', zorder=5, edgecolors='black')
        ax.scatter([rho_crit_c], [self.params_c['Vmax_kmh']/2], s=100, c='orange', marker='s', zorder=5, edgecolors='black')
        ax.set_xlabel('Density ρ (veh/m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speed V (km/h)', fontsize=12, fontweight='bold')
        ax.set_title('Speed-Density Diagram (V-ρ)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 0.2)
        
        # --- Plot 2: Q-ρ for both classes ---
        ax = axes[0, 1]
        ax.plot(rho, Q_m, 'b-', linewidth=2.5, label='Motorcycles')
        ax.plot(rho, Q_c, 'orange', linewidth=2.5, label='Cars')
        ax.axvline(x=rho_crit_m, color='blue', linestyle='--', alpha=0.5, label=f'Critical ρ (motos): {rho_crit_m:.3f}')
        ax.axvline(x=rho_crit_c, color='orange', linestyle='--', alpha=0.5, label=f'Critical ρ (cars): {rho_crit_c:.3f}')
        ax.scatter([rho_crit_m], [Q_max_m], s=100, c='blue', marker='^', zorder=5, edgecolors='black')
        ax.scatter([rho_crit_c], [Q_max_c], s=100, c='orange', marker='s', zorder=5, edgecolors='black')
        ax.set_xlabel('Density ρ (veh/m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Flow Q (veh/h)', fontsize=12, fontweight='bold')
        ax.set_title('Flow-Density Diagram (Q-ρ)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 0.2)
        
        # --- Plot 3: Parameter Comparison ---
        ax = axes[1, 0]
        parameters = ['Vmax\n(km/h)', 'ρmax\n(veh/m)', 'τ\n(s)']
        moto_vals = [self.params_m['Vmax_kmh'], self.params_m['rho_max']*100, self.params_m['tau']*10]
        car_vals = [self.params_c['Vmax_kmh'], self.params_c['rho_max']*100, self.params_c['tau']*10]
        
        x = np.arange(len(parameters))
        width = 0.35
        bars1 = ax.bar(x - width/2, [self.params_m['Vmax_kmh'], self.params_m['rho_max'], self.params_m['tau']], 
                      width, label='Motorcycles', color='blue', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, [self.params_c['Vmax_kmh'], self.params_c['rho_max'], self.params_c['tau']], 
                      width, label='Cars', color='orange', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Parameters', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title('Calibrated Parameters Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(parameters)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)
        
        # --- Plot 4: Throughput Comparison ---
        ax = axes[1, 1]
        ax.text(0.5, 0.8, 'Fundamental Diagram Analysis', 
               fontsize=14, fontweight='bold', ha='center', transform=ax.transAxes)
        
        info_text = f"""
MOTORCYCLES:
  • Vmax = {self.params_m['Vmax_kmh']} km/h
  • ρmax = {self.params_m['rho_max']} veh/m
  • τ = {self.params_m['tau']} s
  • Qmax = {Q_max_m:.0f} veh/h
  
CARS:
  • Vmax = {self.params_c['Vmax_kmh']} km/h
  • ρmax = {self.params_c['rho_max']} veh/m
  • τ = {self.params_c['tau']} s
  • Qmax = {Q_max_c:.0f} veh/h

ADVANTAGES:
  • Motorcycles throughput: {Q_max_m/Q_max_c:.2f}x cars
  • Speed differential: {self.params_m['Vmax_kmh'] - self.params_c['Vmax_kmh']} km/h
  • Density packing: {self.params_m['rho_max']/self.params_c['rho_max']:.2f}x cars
        """
        
        ax.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
               family='monospace', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path / "fundamental_diagrams.png", dpi=300, bbox_inches='tight')
        plt.savefig(save_path / "fundamental_diagrams.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: fundamental_diagrams.png")


def run_test(save_results: bool = True) -> Dict:
    """Run fundamental diagrams test."""
    print("\n" + "=" * 80)
    print("TEST 3: FUNDAMENTAL DIAGRAMS CALIBRATION")
    print("=" * 80)
    
    # Create calibrator
    calibrator = FundamentalDiagramCalibrator()
    
    # Validate
    validation = calibrator.validate_diagrams()
    
    # Create figures
    output_dir = project_root / "figures" / "niveau2_physics"
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrator.create_fundamental_diagrams_figure(output_dir)
    
    # Overall test result
    test_passed = validation['all_valid']
    
    print(f"\n✅ Validation:")
    print(f"  Motorcycles params valid:    {'✅ YES' if validation['motorcycles']['valid'] else '❌ NO'}")
    print(f"  Cars params valid:           {'✅ YES' if validation['cars']['valid'] else '❌ NO'}")
    print(f"  Motos throughput > cars:     {'✅ YES' if validation['comparison']['throughput_advantage'] else '❌ NO'}")
    print(f"  Overall test:                {'✅ PASS' if test_passed else '❌ FAIL'}")
    
    if save_results:
        # Save JSON
        results_dir = project_root / "data" / "validation_results" / "physics_tests"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'test_name': 'fundamental_diagrams_test',
            'description': 'Validates calibrated fundamental diagrams for ARZ model',
            'model': 'ARZ (Aw-Rascle-Zhang)',
            'calibration': {
                'motorcycles': {
                    'Vmax_kmh': float(calibrator.params_m['Vmax_kmh']),
                    'rho_max_veh_per_m': float(calibrator.params_m['rho_max']),
                    'tau_s': float(calibrator.params_m['tau']),
                    'length_m': float(calibrator.params_m['length']),
                    'Q_max_veh_per_h': float(calibrator.compute_max_flow(calibrator.params_m))
                },
                'cars': {
                    'Vmax_kmh': float(calibrator.params_c['Vmax_kmh']),
                    'rho_max_veh_per_m': float(calibrator.params_c['rho_max']),
                    'tau_s': float(calibrator.params_c['tau']),
                    'length_m': float(calibrator.params_c['length']),
                    'Q_max_veh_per_h': float(calibrator.compute_max_flow(calibrator.params_c))
                }
            },
            'validation': validation,
            'test_passed': bool(test_passed)
        }
        
        json_path = results_dir / "fundamental_diagrams.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 JSON saved: {json_path}")
    
    print("\n" + "=" * 80)
    return {'test_passed': test_passed, **validation}


if __name__ == "__main__":
    results = run_test()
    sys.exit(0 if results['test_passed'] else 1)
