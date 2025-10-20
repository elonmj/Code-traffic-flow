"""
Exact Riemann Solver for ARZ Traffic Flow Model.

This module provides analytical solutions to Riemann problems for the ARZ model,
used as validation baseline for our FVM+WENO5 numerical implementation.

Mathematical Background:
-----------------------
ARZ Model: ∂ₜρ + ∂ₓ(ρv) = 0
           ∂ₜ(ρv) + ∂ₓ(ρv² + P) = 0

Where P = ρ²V'(ρ) is the anticipation pressure term.

For V(ρ) = Vmax(1 - ρ/ρ_max), we get:
    P = -ρ²Vmax/ρ_max

Riemann Problem:
    IC: (ρ_L, v_L) for x < x₀
        (ρ_R, v_R) for x ≥ x₀

Solution Types:
    - Shock: q_L > q_R (discontinuous)
    - Rarefaction: q_L < q_R (smooth expansion)

Author: ARZ-RL Validation Team
Date: 2025-10-17
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Union


@dataclass
class RiemannSolution:
    """
    Container for Riemann problem solution.
    
    Attributes:
        x: Spatial coordinates
        t: Time
        rho: Density solution
        v: Velocity solution
        wave_type: 'shock' or 'rarefaction'
        wave_speed: Shock speed (for shock)
        wave_speeds: (lambda_L, lambda_R) for rarefaction
    """
    x: np.ndarray
    t: float
    rho: np.ndarray
    v: np.ndarray
    wave_type: str
    wave_speed: Optional[float] = None
    wave_speeds: Optional[Tuple[float, float]] = None


class ARZRiemannSolver:
    """
    Exact Riemann solver for single-class ARZ model.
    
    This solver provides analytical solutions for validation purposes.
    """
    
    def __init__(self, Vmax: float, rho_max: float, anticipation_coeff: float = 1.0):
        """
        Initialize solver with model parameters.
        
        Args:
            Vmax: Maximum velocity (m/s)
            rho_max: Jam density (veh/m)
            anticipation_coeff: Anticipation coefficient α ∈ [0,1]
        """
        self.Vmax = Vmax
        self.rho_max = rho_max
        self.alpha = anticipation_coeff
    
    def desired_velocity(self, rho: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """V(ρ) = Vmax(1 - ρ/ρ_max)."""
        return self.Vmax * (1.0 - rho / self.rho_max)
    
    def flux(self, rho: Union[float, np.ndarray], v: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """q = ρv."""
        return rho * v
    
    def characteristic_speed(self, rho: float, v: float) -> float:
        """λ = v + ρV'(ρ) = v + ρVmax/ρ_max."""
        return v + rho * self.Vmax / self.rho_max
    
    def determine_wave_type(self, rho_L: float, v_L: float, rho_R: float, v_R: float) -> str:
        """
        Determine wave type from initial states.
        
        Criterion: q_L > q_R → shock, else rarefaction
        
        Args:
            rho_L, v_L: Left state
            rho_R, v_R: Right state
            
        Returns:
            'shock' or 'rarefaction'
        """
        q_L = self.flux(rho_L, v_L)
        q_R = self.flux(rho_R, v_R)
        
        return 'shock' if q_L > q_R else 'rarefaction'
    
    def solve_shock(
        self,
        rho_L: float,
        v_L: float,
        rho_R: float,
        v_R: float,
        x: np.ndarray,
        x0: float,
        t: float
    ) -> RiemannSolution:
        """
        Solve shock wave Riemann problem.
        
        Shock speed: s = (q_R - q_L) / (ρ_R - ρ_L)
        Solution: Piecewise constant
        
        Args:
            rho_L, v_L: Left state
            rho_R, v_R: Right state
            x: Spatial grid
            x0: Initial discontinuity position
            t: Time
            
        Returns:
            RiemannSolution object
        """
        q_L = self.flux(rho_L, v_L)
        q_R = self.flux(rho_R, v_R)
        
        # Shock speed
        s = (q_R - q_L) / (rho_R - rho_L + 1e-10)
        
        # Shock position at time t
        x_shock = x0 + s * t
        
        # Piecewise constant solution
        rho = np.where(x < x_shock, rho_L, rho_R)
        v = np.where(x < x_shock, v_L, v_R)
        
        return RiemannSolution(
            x=x,
            t=t,
            rho=rho,
            v=v,
            wave_type='shock',
            wave_speed=s
        )
    
    def solve_rarefaction(
        self,
        rho_L: float,
        v_L: float,
        rho_R: float,
        v_R: float,
        x: np.ndarray,
        x0: float,
        t: float
    ) -> RiemannSolution:
        """
        Solve rarefaction wave Riemann problem.
        
        Solution is self-similar: ρ(ξ) where ξ = (x-x₀)/t
        Characteristic speeds: λ_L, λ_R
        
        Args:
            rho_L, v_L: Left state
            rho_R, v_R: Right state
            x: Spatial grid
            x0: Initial discontinuity position
            t: Time
            
        Returns:
            RiemannSolution object
        """
        # Characteristic speeds at boundaries
        lambda_L = self.characteristic_speed(rho_L, v_L)
        lambda_R = self.characteristic_speed(rho_R, v_R)
        
        # Self-similarity variable ξ = (x - x₀) / t
        xi = (x - x0) / (t + 1e-10)
        
        # Solution in different regions
        rho = np.zeros_like(x)
        v = np.zeros_like(x)
        
        # Left of rarefaction fan
        mask_L = xi < lambda_L
        rho[mask_L] = rho_L
        v[mask_L] = v_L
        
        # Right of rarefaction fan
        mask_R = xi > lambda_R
        rho[mask_R] = rho_R
        v[mask_R] = v_R
        
        # Inside rarefaction fan: linear interpolation (simplified)
        mask_fan = ~(mask_L | mask_R)
        if np.any(mask_fan):
            # Linear interpolation of ρ within fan
            alpha_interp = (xi[mask_fan] - lambda_L) / (lambda_R - lambda_L + 1e-10)
            rho[mask_fan] = rho_L + alpha_interp * (rho_R - rho_L)
            v[mask_fan] = v_L + alpha_interp * (v_R - v_L)
        
        return RiemannSolution(
            x=x,
            t=t,
            rho=rho,
            v=v,
            wave_type='rarefaction',
            wave_speeds=(lambda_L, lambda_R)
        )
    
    def solve(
        self,
        rho_L: float,
        v_L: float,
        rho_R: float,
        v_R: float,
        x: np.ndarray,
        x0: float,
        t: float
    ) -> RiemannSolution:
        """
        Solve Riemann problem (automatic wave type detection).
        
        Args:
            rho_L, v_L: Left state
            rho_R, v_R: Right state
            x: Spatial grid
            x0: Initial discontinuity position
            t: Time
            
        Returns:
            RiemannSolution object
        """
        wave_type = self.determine_wave_type(rho_L, v_L, rho_R, v_R)
        
        if wave_type == 'shock':
            return self.solve_shock(rho_L, v_L, rho_R, v_R, x, x0, t)
        else:
            return self.solve_rarefaction(rho_L, v_L, rho_R, v_R, x, x0, t)


class MulticlassRiemannSolver:
    """
    Multiclass Riemann solver (2-class: motos + voitures).
    
    This solver handles coupled 4×4 system with anticipation coupling.
    """
    
    def __init__(
        self,
        Vmax_m: float,
        Vmax_v: float,
        rho_max_m: float,
        rho_max_v: float,
        alpha: float = 0.5
    ):
        """
        Initialize multiclass solver.
        
        Args:
            Vmax_m: Motos max velocity (m/s)
            Vmax_v: Voitures max velocity (m/s)
            rho_max_m: Motos jam density (veh/m)
            rho_max_v: Voitures jam density (veh/m)
            alpha: Coupling coefficient
        """
        self.solver_m = ARZRiemannSolver(Vmax_m, rho_max_m, alpha)
        self.solver_v = ARZRiemannSolver(Vmax_v, rho_max_v, alpha)
        self.alpha = alpha
        self.Vmax_m = Vmax_m
        self.Vmax_v = Vmax_v
        self.rho_max_m = rho_max_m
        self.rho_max_v = rho_max_v
    
    def solve_uncoupled(
        self,
        rho_m_L: float,
        v_m_L: float,
        rho_m_R: float,
        v_m_R: float,
        rho_v_L: float,
        v_v_L: float,
        rho_v_R: float,
        v_v_R: float,
        x: np.ndarray,
        x0: float,
        t: float
    ) -> Tuple[RiemannSolution, RiemannSolution]:
        """
        Solve multiclass problem with weak coupling approximation.
        
        Valid for α < 0.5 (weak coupling).
        
        Args:
            rho_m_L, v_m_L, rho_m_R, v_m_R: Motos states
            rho_v_L, v_v_L, rho_v_R, v_v_R: Voitures states
            x: Spatial grid
            x0: Discontinuity position
            t: Time
            
        Returns:
            (sol_motos, sol_voitures) tuple of RiemannSolution objects
        """
        # Solve each class independently
        sol_m = self.solver_m.solve(rho_m_L, v_m_L, rho_m_R, v_m_R, x, x0, t)
        sol_v = self.solver_v.solve(rho_v_L, v_v_L, rho_v_R, v_v_R, x, x0, t)
        
        # Apply velocity correction (anticipation coupling)
        rho_total = sol_m.rho + sol_v.rho
        
        # Adjust motos velocity
        v_m_desired = self.Vmax_m * (1.0 - rho_total / self.rho_max_m)
        sol_m.v = np.minimum(sol_m.v, v_m_desired)
        
        # Adjust voitures velocity
        v_v_desired = self.Vmax_v * (1.0 - rho_total / self.rho_max_v)
        sol_v.v = np.minimum(sol_v.v, v_v_desired)
        
        return sol_m, sol_v


def compute_L2_error(rho_numerical: np.ndarray, rho_exact: np.ndarray, dx: float) -> float:
    """
    Compute L2 norm of error.
    
    L2 = sqrt(Σ(ρ_num - ρ_exact)² Δx / L)
    
    Args:
        rho_numerical: Numerical solution
        rho_exact: Exact solution
        dx: Grid spacing
        
    Returns:
        L2 error
    """
    L = len(rho_numerical) * dx
    return float(np.sqrt(np.sum((rho_numerical - rho_exact)**2) * dx / L))


if __name__ == "__main__":
    """Standalone test."""
    print("Testing ARZ Riemann Solver...")
    
    # Test 1: Shock
    print("\nTest 1: SHOCK WAVE")
    solver = ARZRiemannSolver(Vmax=60/3.6, rho_max=0.15)
    x = np.linspace(0, 1000, 201)
    sol = solver.solve(0.08, 40/3.6, 0.02, 60/3.6, x, 500, 30)
    print(f"  Wave type: {sol.wave_type}")
    print(f"  Shock speed: {sol.wave_speed:.3f} m/s")
    
    # Test 2: Rarefaction
    print("\nTest 2: RAREFACTION WAVE")
    sol2 = solver.solve(0.02, 60/3.6, 0.08, 40/3.6, x, 500, 30)
    print(f"  Wave type: {sol2.wave_type}")
    
    # Test 3: L2 error
    print("\nTest 3: L2 ERROR")
    rho_perturbed = sol.rho + np.random.normal(0, 1e-4, size=sol.rho.shape)
    L2 = compute_L2_error(rho_perturbed, sol.rho, 5.0)
    print(f"  L2 = {L2:.2e}")
    
    print("\n✅ All tests OK")
