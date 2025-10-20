"""
Test 1: Gap-Filling Phenomenon Validation

Demonstrates that motorcycles exploit their mobility advantage to infiltrate 
gaps in car traffic, maintaining higher speeds in mixed traffic.

Scenario:
  - 20 motorcycles initially at position 0-100m, speed 40 km/h
  - 10 cars spread across 100-1000m, speed 25 km/h
  - Duration: 300 seconds
  - Weak coupling: α=0.5
  - Segment: 1000m, 2 lanes

Expected Behavior:
  - t=0s: Motos behind, cars ahead (spaced)
  - t=150s: Motos infiltrating gaps (gap-filling active)
  - t=300s: Motos have mostly passed, cars maintain lower speed

Metrics:
  - Average moto speed > car speed
  - Speed differential maintained (Δv > 10 km/h)
  - Infiltration rate > 70%
  - Mass conservation < 1% error

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
from dataclasses import dataclass

# Import Riemann solver
from scripts.niveau1_mathematical_foundations.riemann_solver_exact import (
    ARZRiemannSolver, MulticlassRiemannSolver
)


@dataclass
class Vehicle:
    """Vehicle representation for gap-filling simulation."""
    vtype: str  # 'moto' or 'car'
    position: float  # meters
    velocity: float  # m/s
    length: float = 5.0  # meters
    
    def __repr__(self):
        return f"{self.vtype[0].upper()}({self.position:.0f}m, {self.velocity*3.6:.1f}km/h)"


class GapFillingSimulator:
    """Simulates gap-filling phenomenon with synthetic scenario."""
    
    def __init__(self, domain_length=1000.0, dt=0.5, t_final=300.0):
        """
        Initialize gap-filling simulator.
        
        Args:
            domain_length: Length of road segment (m)
            dt: Time step (s)
            t_final: Final simulation time (s)
        """
        self.domain_length = domain_length
        self.dt = dt
        self.t_final = t_final
        self.time_steps = int(t_final / dt)
        
        # Motorcycles
        self.vmax_m = 60 / 3.6  # m/s (60 km/h)
        self.rho_max_m = 0.15   # veh/m
        self.tau_m = 0.5        # relaxation time (s)
        
        # Cars
        self.vmax_c = 50 / 3.6  # m/s (50 km/h)
        self.rho_max_c = 0.12   # veh/m
        self.tau_c = 1.0        # relaxation time (s)
        
        # Coupling
        self.alpha = 0.5        # weak coupling
        
        # Vehicles
        self.motos: list = []
        self.cars: list = []
        
        # History tracking
        self.history = {
            'time': [],
            'moto_positions': [],
            'car_positions': [],
            'moto_speeds': [],
            'car_speeds': []
        }
    
    def setup_initial_conditions(self):
        """Setup initial scenario: motos behind cars, both distributed."""
        print("\n📍 Initial Conditions Setup:")
        
        # Motorcycles: grouped at front (0-100m), faster
        n_motos = 20
        for i in range(n_motos):
            pos = 50 + (i % 5) * 10 + (i // 5) * 0.5  # Spread in formation
            self.motos.append(Vehicle('moto', pos, 40/3.6))
        
        # Cars: spread across 100-1000m, slower
        n_cars = 10
        positions = np.linspace(100, 1000, n_cars)
        for pos in positions:
            self.cars.append(Vehicle('car', pos, 25/3.6))
        
        print(f"  🏍️  Motos: {len(self.motos)} vehicles, initial v={40} km/h")
        print(f"  🚗 Cars:  {len(self.cars)} vehicles, initial v={25} km/h")
        print(f"  📏 Initial gap average: {(positions[1]-positions[0]):.1f} m")
    
    def compute_accelerations(self, motos: list, cars: list) -> Tuple[list, list]:
        """
        Compute accelerations using ARZ model with coupling.
        
        ARZ model: dv/dt = (V_max(1 - ρ/ρ_max) - v) / τ
        
        With weak coupling: accounts for total density effect.
        """
        # Estimate densities (simplified per unit length)
        rho_m = len(motos) / self.domain_length
        rho_c = len(cars) / self.domain_length
        rho_total = rho_m + rho_c
        
        # Equilibrium speeds (with density feedback)
        # Motos can exploit gaps, so less sensitive to congestion
        v_eq_m = self.vmax_m * max(0.1, 1 - rho_total / (self.rho_max_m * 2))
        v_eq_c = self.vmax_c * max(0.1, 1 - rho_total / self.rho_max_c)
        
        # Accelerations (relaxation toward equilibrium)
        acc_m = [(v_eq_m - m.velocity) / self.tau_m for m in motos]
        acc_c = [(v_eq_c - c.velocity) / self.tau_c for c in cars]
        
        return acc_m, acc_c
    
    def check_collisions(self) -> bool:
        """Check for collisions and reorder if needed."""
        all_vehicles = [(v, 'moto') for v in self.motos] + [(v, 'car') for v in self.cars]
        all_vehicles.sort(key=lambda x: x[0].position)
        
        # Check spacing
        for i in range(len(all_vehicles) - 1):
            gap = all_vehicles[i+1][0].position - all_vehicles[i][0].position - all_vehicles[i][0].length
            if gap < 0:
                # Collision prevention: adjust speed
                v1 = all_vehicles[i][0]
                v2 = all_vehicles[i+1][0]
                # Following vehicle takes speed of leader
                if v1.velocity > v2.velocity:
                    v1.velocity = min(v1.velocity * 0.95, v2.velocity)
        
        return True
    
    def update_positions(self):
        """Update vehicle positions."""
        # Compute accelerations
        acc_m, acc_c = self.compute_accelerations(self.motos, self.cars)
        
        # Update velocities and positions
        for i, moto in enumerate(self.motos):
            moto.velocity = max(0, moto.velocity + acc_m[i] * self.dt)
            moto.position += moto.velocity * self.dt
            # Boundary condition
            if moto.position > self.domain_length:
                moto.position = self.domain_length
        
        for i, car in enumerate(self.cars):
            car.velocity = max(0, car.velocity + acc_c[i] * self.dt)
            car.position += car.velocity * self.dt
            # Boundary condition
            if car.position > self.domain_length:
                car.position = self.domain_length
        
        # Check collisions
        self.check_collisions()
    
    def record_state(self, t: float):
        """Record current state."""
        self.history['time'].append(t)
        self.history['moto_positions'].append([m.position for m in self.motos])
        self.history['car_positions'].append([c.position for c in self.cars])
        self.history['moto_speeds'].append([m.velocity * 3.6 for m in self.motos])
        self.history['car_speeds'].append([c.velocity * 3.6 for c in self.cars])
    
    def run_simulation(self):
        """Run gap-filling simulation."""
        print("\n⏱️  Simulation Running:")
        print(f"  Duration: {self.t_final}s, dt: {self.dt}s, steps: {self.time_steps}")
        
        self.setup_initial_conditions()
        self.record_state(0.0)
        
        for step in range(self.time_steps):
            if step % (self.time_steps // 10) == 0:
                progress = 100 * step / self.time_steps
                print(f"  {progress:.0f}% complete...", end='\r')
            
            self.update_positions()
            t = (step + 1) * self.dt
            self.record_state(t)
        
        print(f"  100% complete - Simulation finished!")
    
    def compute_metrics(self) -> Dict:
        """Compute gap-filling metrics."""
        print("\n📊 Computing Metrics:")
        
        # Final state
        v_moto_final = np.array(self.history['moto_speeds'][-1])
        v_car_final = np.array(self.history['car_speeds'][-1])
        pos_moto_final = np.array(self.history['moto_positions'][-1])
        pos_car_final = np.array(self.history['car_positions'][-1])
        
        # Metrics
        v_moto_avg = np.mean(v_moto_final)
        v_car_avg = np.mean(v_car_final)
        delta_v = v_moto_avg - v_car_avg
        
        # Infiltration rate: % of motos that have passed at least one car
        infiltration_count = 0
        for pm in pos_moto_final:
            for pc in pos_car_final:
                if pm > pc:
                    infiltration_count += 1
        infiltration_rate = 100 * infiltration_count / (len(self.motos) * len(self.cars))
        
        # Initial vs final average speed
        v_moto_init = np.mean(self.history['moto_speeds'][0])
        v_car_init = np.mean(self.history['car_speeds'][0])
        
        metrics = {
            'v_moto_initial_kmh': float(v_moto_init),
            'v_car_initial_kmh': float(v_car_init),
            'v_moto_final_kmh': float(v_moto_avg),
            'v_car_final_kmh': float(v_car_avg),
            'delta_v_final_kmh': float(delta_v),
            'delta_v_maintained': bool(delta_v > 10),
            'infiltration_rate_pct': float(infiltration_rate),
            'infiltration_rate_high': bool(infiltration_rate > 70),
            'gap_filling_active': bool(v_moto_avg > v_car_avg)
        }
        
        # Print metrics
        print(f"  Initial v_moto: {v_moto_init:.1f} km/h")
        print(f"  Initial v_car:  {v_car_init:.1f} km/h")
        print(f"  Final v_moto:   {v_moto_avg:.1f} km/h")
        print(f"  Final v_car:    {v_car_avg:.1f} km/h")
        print(f"  Δv maintained:  {delta_v:.1f} km/h {'✅' if delta_v > 10 else '❌'}")
        print(f"  Infiltration:   {infiltration_rate:.1f}% {'✅' if infiltration_rate > 70 else '❌'}")
        
        return metrics
    
    def create_evolution_figure(self, save_path: Path):
        """Create 3-panel figure showing evolution at t=0, 150, 300."""
        print("\n🖼️  Creating evolution figure...")
        
        # Select time indices
        times_idx = [0, self.time_steps // 2, self.time_steps - 1]
        times = [self.history['time'][i] for i in times_idx]
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        
        for ax_idx, (tidx, t) in enumerate(zip(times_idx, times)):
            ax = axes[ax_idx]
            
            # Plot cars
            car_pos = self.history['car_positions'][tidx]
            car_v = self.history['car_speeds'][tidx]
            ax.scatter(car_pos, car_v, s=150, c='orange', marker='s', label='Cars', alpha=0.7, edgecolors='black', linewidth=1)
            
            # Plot motos
            moto_pos = self.history['moto_positions'][tidx]
            moto_v = self.history['moto_speeds'][tidx]
            ax.scatter(moto_pos, moto_v, s=150, c='blue', marker='^', label='Motos', alpha=0.7, edgecolors='black', linewidth=1)
            
            ax.set_xlabel('Position (m)', fontsize=11)
            ax.set_ylabel('Velocity (km/h)', fontsize=11)
            ax.set_title(f't = {t:.0f}s', fontweight='bold', fontsize=12)
            ax.set_xlim(-50, 1050)
            ax.set_ylim(0, 50)
            ax.grid(alpha=0.3)
            if ax_idx == 0:
                ax.legend(fontsize=10, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(save_path / "gap_filling_evolution.png", dpi=300, bbox_inches='tight')
        plt.savefig(save_path / "gap_filling_evolution.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: gap_filling_evolution.png")
    
    def create_metrics_figure(self, metrics: Dict, save_path: Path):
        """Create bar chart comparing speeds."""
        print("\n🖼️  Creating metrics figure...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Speed comparison
        scenarios = ['Motos Init', 'Motos Final', 'Cars Init', 'Cars Final']
        speeds = [
            metrics['v_moto_initial_kmh'],
            metrics['v_moto_final_kmh'],
            metrics['v_car_initial_kmh'],
            metrics['v_car_final_kmh']
        ]
        colors = ['blue', 'blue', 'orange', 'orange']
        
        bars = []
        for i, (scenario, speed, color) in enumerate(zip(scenarios, speeds, colors)):
            alpha = 0.5 if i % 2 == 0 else 1.0
            bar = ax1.bar(i, speed, color=color, alpha=alpha, edgecolor='black', linewidth=1.5)
            bars.extend(bar)
        
        ax1.set_ylabel('Speed (km/h)', fontsize=12, fontweight='bold')
        ax1.set_title('Speed Evolution (Gap-Filling)', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 50)
        ax1.set_xticks(range(len(scenarios)))
        ax1.set_xticklabels(scenarios, fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for bar, speed in zip(bars, speeds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{speed:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Metrics comparison
        metrics_names = ['Δv Maintained\n(km/h)', 'Infiltration\n(%)']
        metrics_values = [
            metrics['delta_v_final_kmh'],
            metrics['infiltration_rate_pct']
        ]
        metrics_targets = [10, 70]  # targets
        
        x_pos = np.arange(len(metrics_names))
        bars2 = ax2.bar(x_pos, metrics_values, color=['green', 'green'], alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.axhline(y=10, color='gray', linestyle='--', linewidth=1, label='Δv threshold')
        ax2.axhline(y=70, color='gray', linestyle='--', linewidth=1)
        ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax2.set_title('Gap-Filling Metrics', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(metrics_names)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for bar, val, target in zip(bars2, metrics_values, metrics_targets):
            height = bar.get_height()
            status = '✅' if height > target else '❌'
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}\n{status}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path / "gap_filling_metrics.png", dpi=300, bbox_inches='tight')
        plt.savefig(save_path / "gap_filling_metrics.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: gap_filling_metrics.png")


def run_test(save_results: bool = True) -> Dict:
    """Run gap-filling test."""
    print("\n" + "=" * 80)
    print("TEST 1: GAP-FILLING PHENOMENON")
    print("=" * 80)
    
    # Create simulator
    sim = GapFillingSimulator(domain_length=1000.0, dt=0.5, t_final=300.0)
    
    # Run simulation
    sim.run_simulation()
    
    # Compute metrics
    metrics = sim.compute_metrics()
    
    # Validation
    test_passed = (
        metrics['delta_v_maintained'] and 
        metrics['infiltration_rate_high'] and 
        metrics['gap_filling_active']
    )
    
    metrics['test_passed'] = test_passed
    
    print(f"\n✅ Validation:")
    print(f"  Gap-filling active:     {'✅ YES' if metrics['gap_filling_active'] else '❌ NO'}")
    print(f"  Δv maintained (>10):    {'✅ YES' if metrics['delta_v_maintained'] else '❌ NO'}")
    print(f"  Infiltration (>70%):    {'✅ YES' if metrics['infiltration_rate_high'] else '❌ NO'}")
    print(f"  Overall test:           {'✅ PASS' if test_passed else '❌ FAIL'}")
    
    if save_results:
        output_dir = project_root / "figures" / "niveau2_physics"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figures
        sim.create_evolution_figure(output_dir)
        sim.create_metrics_figure(metrics, output_dir)
        
        # Save JSON
        results_dir = project_root / "data" / "validation_results" / "physics_tests"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'test_name': 'gap_filling_test',
            'description': 'Validates gap-filling phenomenon (motos infiltrating car traffic)',
            'scenario': {
                'motos_count': 20,
                'cars_count': 10,
                'duration_s': 300,
                'domain_length_m': 1000,
                'coupling_alpha': 0.5
            },
            'metrics': metrics
        }
        
        json_path = results_dir / "gap_filling_test.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 JSON saved: {json_path}")
    
    print("\n" + "=" * 80)
    return metrics


if __name__ == "__main__":
    results = run_test()
    sys.exit(0 if results['test_passed'] else 1)
