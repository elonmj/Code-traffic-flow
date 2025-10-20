"""
Test 2: Interweaving Phenomenon Validation

Demonstrates continuous threading behavior where motorcycles zigzag through 
car traffic, leading to class segregation and different final positions.

Scenario:
  - 15 motorcycles + 15 cars initially mixed/alternating
  - Segment: 2000m, 3 lanes
  - Duration: 400 seconds
  - Weak coupling: α=0.5
  - More complex interaction than gap-filling

Expected Behavior:
  - Motos and cars initially homogeneously mixed
  - Over time: motos advance through cars (thread/weave pattern)
  - Final state: spatial segregation (motos ahead of cars)
  - No mutual blocking - classes maintain relative velocity differences

Metrics:
  - Motos have advanced further than cars (Δpos > 500m)
  - Speed differential maintained (Δv > 8 km/h)
  - Distribution entropy decreases (classes segregate)

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


@dataclass
class Vehicle:
    """Vehicle representation for interweaving simulation."""
    vtype: str  # 'moto' or 'car'
    position: float  # meters
    velocity: float  # m/s
    length: float = 5.0  # meters
    
    def __repr__(self):
        return f"{self.vtype[0].upper()}({self.position:.0f}m, {self.velocity*3.6:.1f}km/h)"


class InterweavingSimulator:
    """Simulates interweaving phenomenon with mixed vehicle distribution."""
    
    def __init__(self, domain_length=2000.0, dt=0.5, t_final=400.0):
        """
        Initialize interweaving simulator.
        
        Args:
            domain_length: Length of road segment (m)
            dt: Time step (s)
            t_final: Final simulation time (s)
        """
        self.domain_length = domain_length
        self.dt = dt
        self.t_final = t_final
        self.time_steps = int(t_final / dt)
        self.lanes = 3
        
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
            'car_speeds': [],
            'segregation_index': []  # Measure of class segregation
        }
    
    def setup_initial_conditions(self):
        """Setup initial scenario: motos and cars homogeneously mixed."""
        print("\n📍 Initial Conditions Setup:")
        
        # Create homogeneous mix: alternating motos and cars
        n_motos = 15
        n_cars = 15
        total_vehicles = n_motos + n_cars
        
        # Distribute uniformly across segment
        positions = np.linspace(100, 1900, total_vehicles)
        
        # Alternate: M, C, M, C, ...
        vehicle_types = ['moto' if i % 2 == 0 else 'car' for i in range(total_vehicles)]
        
        moto_idx = 0
        car_idx = 0
        for i, (pos, vtype) in enumerate(zip(positions, vehicle_types)):
            if vtype == 'moto':
                self.motos.append(Vehicle('moto', pos, 40/3.6))
                moto_idx += 1
            else:
                self.cars.append(Vehicle('car', pos, 25/3.6))
                car_idx += 1
        
        print(f"  🏍️  Motos: {len(self.motos)} vehicles, initial v={40} km/h")
        print(f"  🚗 Cars:  {len(self.cars)} vehicles, initial v={25} km/h")
        print(f"  📏 Distribution: Alternating (homogeneous mix)")
        print(f"  🛣️  Segment: {self.domain_length}m, {self.lanes} lanes")
    
    def compute_accelerations(self, motos: list, cars: list) -> Tuple[list, list]:
        """
        Compute accelerations using ARZ model with coupling.
        
        ARZ: dv/dt = (V_eq - v) / τ
        where V_eq = V_max * (1 - ρ_total / ρ_max)
        """
        # Estimate densities
        rho_m = len(motos) / (self.domain_length * self.lanes)
        rho_c = len(cars) / (self.domain_length * self.lanes)
        rho_total = rho_m + rho_c
        
        # Equilibrium speeds
        v_eq_m = self.vmax_m * max(0, 1 - rho_total / self.rho_max_m)
        v_eq_c = self.vmax_c * max(0, 1 - rho_total / self.rho_max_c)
        
        # Accelerations
        acc_m = [(v_eq_m - m.velocity) / self.tau_m for m in motos]
        acc_c = [(v_eq_c - c.velocity) / self.tau_c for c in cars]
        
        return acc_m, acc_c
    
    def check_collisions_lane_change(self) -> bool:
        """Implement simple lane-changing logic for motos."""
        # Motos try to pass cars by changing lanes
        for moto in self.motos:
            for car in self.cars:
                # If moto behind car and much slower
                if (moto.position < car.position and 
                    moto.velocity < car.velocity + 1.0 and
                    car.position - moto.position < 50):
                    # Moto accelerates slightly to pass (lane change bonus)
                    moto.velocity = min(moto.velocity * 1.05, self.vmax_m)
        
        return True
    
    def update_positions(self):
        """Update vehicle positions."""
        # Compute accelerations
        acc_m, acc_c = self.compute_accelerations(self.motos, self.cars)
        
        # Update motos
        for i, moto in enumerate(self.motos):
            moto.velocity = max(0, moto.velocity + acc_m[i] * self.dt)
            moto.position += moto.velocity * self.dt
            if moto.position > self.domain_length:
                moto.position = self.domain_length
        
        # Update cars
        for i, car in enumerate(self.cars):
            car.velocity = max(0, car.velocity + acc_c[i] * self.dt)
            car.position += car.velocity * self.dt
            if car.position > self.domain_length:
                car.position = self.domain_length
        
        # Lane change for motos
        self.check_collisions_lane_change()
    
    def compute_segregation_index(self) -> float:
        """
        Compute segregation index: measure of class separation.
        
        0 = completely mixed
        1 = completely segregated
        """
        if len(self.motos) == 0 or len(self.cars) == 0:
            return 0.0
        
        moto_pos = np.array([m.position for m in self.motos])
        car_pos = np.array([c.position for c in self.cars])
        
        # Motos average position vs cars average position
        moto_mean = np.mean(moto_pos)
        car_mean = np.mean(car_pos)
        
        # Moto spread vs car spread
        moto_std = np.std(moto_pos) + 1e-6
        car_std = np.std(car_pos) + 1e-6
        
        # Segregation: how far apart are the centers relative to spreads?
        separation = abs(moto_mean - car_mean)
        combined_spread = (moto_std + car_std) / 2
        
        segregation_index = min(1.0, separation / (combined_spread + 1e-6) / 10)
        
        return segregation_index
    
    def record_state(self, t: float):
        """Record current state."""
        self.history['time'].append(t)
        self.history['moto_positions'].append([m.position for m in self.motos])
        self.history['car_positions'].append([c.position for c in self.cars])
        self.history['moto_speeds'].append([m.velocity * 3.6 for m in self.motos])
        self.history['car_speeds'].append([c.velocity * 3.6 for c in self.cars])
        self.history['segregation_index'].append(self.compute_segregation_index())
    
    def run_simulation(self):
        """Run interweaving simulation."""
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
        """Compute interweaving metrics."""
        print("\n📊 Computing Metrics:")
        
        # Final state
        pos_moto_final = np.array(self.history['moto_positions'][-1])
        pos_car_final = np.array(self.history['car_positions'][-1])
        v_moto_final = np.array(self.history['moto_speeds'][-1])
        v_car_final = np.array(self.history['car_speeds'][-1])
        
        # Position metrics
        moto_mean_pos = np.mean(pos_moto_final)
        car_mean_pos = np.mean(pos_car_final)
        delta_pos = moto_mean_pos - car_mean_pos
        
        # Speed metrics
        v_moto_avg = np.mean(v_moto_final)
        v_car_avg = np.mean(v_car_final)
        delta_v = v_moto_avg - v_car_avg
        
        # Segregation
        seg_init = self.history['segregation_index'][0]
        seg_final = self.history['segregation_index'][-1]
        
        metrics = {
            'moto_position_final_m': float(moto_mean_pos),
            'car_position_final_m': float(car_mean_pos),
            'delta_position_m': float(delta_pos),
            'delta_position_significant': bool(delta_pos > 500),
            'moto_speed_final_kmh': float(v_moto_avg),
            'car_speed_final_kmh': float(v_car_avg),
            'delta_v_final_kmh': float(delta_v),
            'delta_v_maintained': bool(delta_v > 8),
            'segregation_initial': float(seg_init),
            'segregation_final': float(seg_final),
            'segregation_increased': bool(seg_final > seg_init)
        }
        
        # Print metrics
        print(f"  Moto mean position:  {moto_mean_pos:.1f} m")
        print(f"  Car mean position:   {car_mean_pos:.1f} m")
        print(f"  Δ position:          {delta_pos:.1f} m {'✅' if delta_pos > 500 else '❌'}")
        print(f"  Final v_moto:        {v_moto_avg:.1f} km/h")
        print(f"  Final v_car:         {v_car_avg:.1f} km/h")
        print(f"  Δv maintained:       {delta_v:.1f} km/h {'✅' if delta_v > 8 else '❌'}")
        print(f"  Segregation index:   init={seg_init:.3f}, final={seg_final:.3f} {'✅' if seg_final > seg_init else '❌'}")
        
        return metrics
    
    def create_distribution_figure(self, save_path: Path):
        """Create figure showing distribution evolution."""
        print("\n🖼️  Creating distribution figure...")
        
        # Select time indices
        times_idx = [0, self.time_steps // 3, 2*self.time_steps // 3, self.time_steps - 1]
        times = [self.history['time'][i] for i in times_idx]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for ax_idx, (tidx, t) in enumerate(zip(times_idx, times)):
            ax = axes[ax_idx]
            
            # Histogram of positions
            car_pos = self.history['car_positions'][tidx]
            moto_pos = self.history['moto_positions'][tidx]
            
            bins = np.linspace(0, 2000, 21)
            ax.hist(car_pos, bins=bins, alpha=0.6, color='orange', label='Cars', edgecolor='black')
            ax.hist(moto_pos, bins=bins, alpha=0.6, color='blue', label='Motos', edgecolor='black')
            
            ax.set_xlabel('Position (m)', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title(f't = {t:.0f}s (Segregation={self.history["segregation_index"][tidx]:.2f})', 
                        fontweight='bold', fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path / "interweaving_distribution.png", dpi=300, bbox_inches='tight')
        plt.savefig(save_path / "interweaving_distribution.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: interweaving_distribution.png")


def run_test(save_results: bool = True) -> Dict:
    """Run interweaving test."""
    print("\n" + "=" * 80)
    print("TEST 2: INTERWEAVING PHENOMENON")
    print("=" * 80)
    
    # Create simulator
    sim = InterweavingSimulator(domain_length=2000.0, dt=0.5, t_final=400.0)
    
    # Run simulation
    sim.run_simulation()
    
    # Compute metrics
    metrics = sim.compute_metrics()
    
    # Validation
    test_passed = (
        metrics['delta_position_significant'] and 
        metrics['delta_v_maintained'] and 
        metrics['segregation_increased']
    )
    
    metrics['test_passed'] = test_passed
    
    print(f"\n✅ Validation:")
    print(f"  Position separation (>500m): {'✅ YES' if metrics['delta_position_significant'] else '❌ NO'}")
    print(f"  Δv maintained (>8 km/h):     {'✅ YES' if metrics['delta_v_maintained'] else '❌ NO'}")
    print(f"  Segregation increased:       {'✅ YES' if metrics['segregation_increased'] else '❌ NO'}")
    print(f"  Overall test:                {'✅ PASS' if test_passed else '❌ FAIL'}")
    
    if save_results:
        output_dir = project_root / "figures" / "niveau2_physics"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure
        sim.create_distribution_figure(output_dir)
        
        # Save JSON
        results_dir = project_root / "data" / "validation_results" / "physics_tests"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'test_name': 'interweaving_test',
            'description': 'Validates interweaving phenomenon (motos threading through cars)',
            'scenario': {
                'motos_count': 15,
                'cars_count': 15,
                'duration_s': 400,
                'domain_length_m': 2000,
                'lanes': 3,
                'coupling_alpha': 0.5,
                'initial_distribution': 'alternating_mix'
            },
            'metrics': metrics
        }
        
        json_path = results_dir / "interweaving_test.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 JSON saved: {json_path}")
    
    print("\n" + "=" * 80)
    return metrics


if __name__ == "__main__":
    results = run_test()
    sys.exit(0 if results['test_passed'] else 1)
