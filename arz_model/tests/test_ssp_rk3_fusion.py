"""
Tests de validation pour la fusion des kernels SSP-RK3 (Phase 2.3).

Ce module vérifie que le kernel fusionné produit des résultats numériquement
équivalents aux kernels séparés legacy, tout en offrant de meilleures performances.

Tests inclus:
- Équivalence numérique (tolérance: 1e-12)
- Tests avec différentes tailles de problème
- Tests avec différentes conditions initiales
- Tests de performance comparatifs
"""

import numpy as np
import pytest
from numba import cuda
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import both fused and legacy implementations
from arz_model.numerics.gpu.ssp_rk3_cuda import (
    integrate_ssp_rk3_gpu,
    SSP_RK3_GPU,
    ssp_rk3_fused_kernel,
    ssp_rk3_stage1_kernel,
    ssp_rk3_stage2_kernel,
    ssp_rk3_stage3_kernel,
)


# ================================================================
# Test Fixtures and Utilities
# ================================================================

@pytest.fixture
def sample_state_random():
    """État aléatoire pour tests génériques."""
    N = 256
    num_vars = 4
    np.random.seed(42)
    return np.random.rand(N, num_vars).astype(np.float64)


@pytest.fixture
def sample_state_smooth():
    """État lisse (sinusoïdale) pour tests de convergence."""
    N = 256
    num_vars = 4
    x = np.linspace(0, 2*np.pi, N)
    u = np.zeros((N, num_vars))
    for v in range(num_vars):
        u[:, v] = 0.5 + 0.3 * np.sin((v+1) * x)
    return u.astype(np.float64)


@pytest.fixture
def sample_state_discontinuous():
    """État avec discontinuités (test de robustesse)."""
    N = 256
    num_vars = 4
    u = np.ones((N, num_vars)) * 0.5
    # Discontinuité au milieu
    u[N//2:, :] = 0.8
    return u.astype(np.float64)


def zero_flux_divergence(u_device, flux_div_device):
    """
    Fonction de divergence de flux nulle (placeholder pour tests).
    
    Cette fonction est utilisée pour tester la logique RK3 indépendamment
    du calcul de flux. En pratique, elle sera remplacée par la vraie
    chaîne WENO + Riemann.
    """
    flux_div_device[:, :] = 0.0


def simple_flux_divergence(u_device, flux_div_device):
    """
    Fonction de divergence de flux simple (diffusion linéaire).
    
    Utilisée pour tester avec un opérateur spatial non-trivial.
    L(u) = -u (simple décroissance exponentielle).
    """
    # Copier u_device vers flux_div_device et multiplier par -1
    flux_div_device[:, :] = -u_device[:, :]


# ================================================================
# Tests d'Équivalence Numérique
# ================================================================

class TestNumericalEquivalence:
    """Tests de validation de l'équivalence numérique fused vs legacy."""
    
    def test_equivalence_zero_flux_random(self, sample_state_random):
        """Test avec flux nul et état aléatoire."""
        u0 = sample_state_random
        dt = 0.001
        dx = 0.01
        
        # Legacy (3 kernels séparés)
        u_legacy = integrate_ssp_rk3_gpu(
            u0, dt, dx, 
            compute_flux_divergence_func=zero_flux_divergence,
            use_fused_kernel=False
        )
        
        # Fused (kernel unique)
        u_fused = integrate_ssp_rk3_gpu(
            u0, dt, dx,
            use_fused_kernel=True
        )
        
        # Vérification de l'équivalence (tolérance machine)
        np.testing.assert_allclose(
            u_legacy, u_fused,
            atol=1e-12, rtol=1e-12,
            err_msg="Fused kernel does not match legacy with zero flux"
        )
    
    def test_equivalence_zero_flux_smooth(self, sample_state_smooth):
        """Test avec flux nul et état lisse."""
        u0 = sample_state_smooth
        dt = 0.001
        dx = 0.01
        
        u_legacy = integrate_ssp_rk3_gpu(
            u0, dt, dx,
            compute_flux_divergence_func=zero_flux_divergence,
            use_fused_kernel=False
        )
        
        u_fused = integrate_ssp_rk3_gpu(
            u0, dt, dx,
            use_fused_kernel=True
        )
        
        np.testing.assert_allclose(
            u_legacy, u_fused,
            atol=1e-12, rtol=1e-12,
            err_msg="Fused kernel does not match legacy with smooth state"
        )
    
    def test_equivalence_zero_flux_discontinuous(self, sample_state_discontinuous):
        """Test avec flux nul et discontinuités."""
        u0 = sample_state_discontinuous
        dt = 0.001
        dx = 0.01
        
        u_legacy = integrate_ssp_rk3_gpu(
            u0, dt, dx,
            compute_flux_divergence_func=zero_flux_divergence,
            use_fused_kernel=False
        )
        
        u_fused = integrate_ssp_rk3_gpu(
            u0, dt, dx,
            use_fused_kernel=True
        )
        
        np.testing.assert_allclose(
            u_legacy, u_fused,
            atol=1e-12, rtol=1e-12,
            err_msg="Fused kernel does not match legacy with discontinuities"
        )
    
    @pytest.mark.parametrize("N", [64, 128, 256, 512, 1024])
    def test_equivalence_various_sizes(self, N):
        """Test avec différentes tailles de problème."""
        num_vars = 4
        np.random.seed(42)
        u0 = np.random.rand(N, num_vars).astype(np.float64)
        dt = 0.001
        dx = 1.0 / N
        
        u_legacy = integrate_ssp_rk3_gpu(
            u0, dt, dx,
            compute_flux_divergence_func=zero_flux_divergence,
            use_fused_kernel=False
        )
        
        u_fused = integrate_ssp_rk3_gpu(
            u0, dt, dx,
            use_fused_kernel=True
        )
        
        np.testing.assert_allclose(
            u_legacy, u_fused,
            atol=1e-12, rtol=1e-12,
            err_msg=f"Fused kernel does not match legacy for N={N}"
        )
    
    @pytest.mark.parametrize("dt", [0.0001, 0.001, 0.01, 0.1])
    def test_equivalence_various_dt(self, sample_state_random, dt):
        """Test avec différents pas de temps."""
        u0 = sample_state_random
        dx = 0.01
        
        u_legacy = integrate_ssp_rk3_gpu(
            u0, dt, dx,
            compute_flux_divergence_func=zero_flux_divergence,
            use_fused_kernel=False
        )
        
        u_fused = integrate_ssp_rk3_gpu(
            u0, dt, dx,
            use_fused_kernel=True
        )
        
        np.testing.assert_allclose(
            u_legacy, u_fused,
            atol=1e-12, rtol=1e-12,
            err_msg=f"Fused kernel does not match legacy for dt={dt}"
        )


# ================================================================
# Tests de Conservation des Propriétés Mathématiques
# ================================================================

class TestMathematicalProperties:
    """Tests vérifiant les propriétés mathématiques du schéma SSP-RK3."""
    
    def test_constant_state_preservation_fused(self):
        """Vérifie que le kernel fusionné préserve un état constant (avec flux nul)."""
        N = 256
        num_vars = 4
        u0 = np.ones((N, num_vars)) * 0.5
        dt = 0.01
        dx = 0.01
        
        u_final = integrate_ssp_rk3_gpu(
            u0, dt, dx,
            use_fused_kernel=True
        )
        
        # Avec flux nul, l'état devrait rester constant
        np.testing.assert_allclose(
            u_final, u0,
            atol=1e-14, rtol=1e-14,
            err_msg="Fused kernel does not preserve constant state with zero flux"
        )
    
    def test_linearity_fused(self):
        """Vérifie la linéarité du schéma (avec flux nul)."""
        N = 256
        num_vars = 4
        np.random.seed(42)
        
        u1 = np.random.rand(N, num_vars).astype(np.float64)
        u2 = np.random.rand(N, num_vars).astype(np.float64)
        alpha = 0.7
        beta = 0.3
        
        dt = 0.001
        dx = 0.01
        
        # alpha * u1 + beta * u2
        u_combined = alpha * u1 + beta * u2
        
        # Intégration des états séparément
        u1_final = integrate_ssp_rk3_gpu(u1, dt, dx, use_fused_kernel=True)
        u2_final = integrate_ssp_rk3_gpu(u2, dt, dx, use_fused_kernel=True)
        
        # Intégration de la combinaison
        u_combined_final = integrate_ssp_rk3_gpu(u_combined, dt, dx, use_fused_kernel=True)
        
        # Vérifier la linéarité: L(alpha*u1 + beta*u2) = alpha*L(u1) + beta*L(u2)
        expected = alpha * u1_final + beta * u2_final
        
        np.testing.assert_allclose(
            u_combined_final, expected,
            atol=1e-12, rtol=1e-12,
            err_msg="Fused kernel violates linearity property"
        )


# ================================================================
# Tests de Performance Comparatifs
# ================================================================

class TestPerformance:
    """Tests de performance comparatifs fused vs legacy."""
    
    @pytest.mark.benchmark
    def test_performance_comparison(self, benchmark, sample_state_random):
        """Compare les performances fused vs legacy."""
        u0 = sample_state_random
        dt = 0.001
        dx = 0.01
        
        # Benchmark legacy
        def run_legacy():
            return integrate_ssp_rk3_gpu(
                u0, dt, dx,
                compute_flux_divergence_func=zero_flux_divergence,
                use_fused_kernel=False
            )
        
        # Benchmark fused
        def run_fused():
            return integrate_ssp_rk3_gpu(
                u0, dt, dx,
                use_fused_kernel=True
            )
        
        # Warm-up (compile kernels)
        run_legacy()
        run_fused()
        
        # Measure (simple timing, not using pytest-benchmark)
        import time
        
        # Legacy timing
        t0 = time.perf_counter()
        for _ in range(100):
            run_legacy()
        cuda.synchronize()
        t_legacy = time.perf_counter() - t0
        
        # Fused timing
        t0 = time.perf_counter()
        for _ in range(100):
            run_fused()
        cuda.synchronize()
        t_fused = time.perf_counter() - t0
        
        speedup = t_legacy / t_fused
        print(f"\n🚀 Performance Speedup: {speedup:.2f}x")
        print(f"   Legacy time: {t_legacy*1000:.2f} ms")
        print(f"   Fused time:  {t_fused*1000:.2f} ms")
        
        # Le kernel fusionné devrait être plus rapide
        # (on ne fait pas d'assertion stricte car cela dépend du hardware)
        # Objectif: speedup >= 1.2 (20% faster minimum)
        assert speedup >= 1.0, f"Fused kernel is slower than legacy (speedup={speedup:.2f})"


# ================================================================
# Tests de Robustesse et Edge Cases
# ================================================================

class TestRobustness:
    """Tests de robustesse et cas limites."""
    
    def test_very_small_dt(self, sample_state_random):
        """Test avec un très petit pas de temps."""
        u0 = sample_state_random
        dt = 1e-8
        dx = 0.01
        
        # Ne devrait pas crasher
        u_fused = integrate_ssp_rk3_gpu(u0, dt, dx, use_fused_kernel=True)
        
        # Avec dt très petit et flux nul, l'état devrait rester quasi-constant
        np.testing.assert_allclose(u_fused, u0, atol=1e-7, rtol=1e-7)
    
    def test_large_dt(self, sample_state_random):
        """Test avec un grand pas de temps."""
        u0 = sample_state_random
        dt = 1.0
        dx = 0.01
        
        # Ne devrait pas crasher (même si le schéma peut être instable)
        u_fused = integrate_ssp_rk3_gpu(u0, dt, dx, use_fused_kernel=True)
        
        # Vérifier que les valeurs sont finies
        assert np.all(np.isfinite(u_fused)), "Fused kernel produced NaN/Inf with large dt"
    
    def test_zero_state(self):
        """Test avec un état nul."""
        N = 256
        num_vars = 4
        u0 = np.zeros((N, num_vars))
        dt = 0.01
        dx = 0.01
        
        u_fused = integrate_ssp_rk3_gpu(u0, dt, dx, use_fused_kernel=True)
        
        # L'état devrait rester nul
        np.testing.assert_allclose(u_fused, u0, atol=1e-14)
    
    def test_single_cell(self):
        """Test avec une seule cellule (cas trivial)."""
        N = 1
        num_vars = 4
        u0 = np.array([[0.5, 0.6, 0.7, 0.8]])
        dt = 0.01
        dx = 0.01
        
        # Ne devrait pas crasher
        u_fused = integrate_ssp_rk3_gpu(u0, dt, dx, use_fused_kernel=True)
        
        assert u_fused.shape == u0.shape


# ================================================================
# Tests d'Intégration
# ================================================================

class TestIntegration:
    """Tests d'intégration avec le reste du système."""
    
    def test_multiple_timesteps_consistency(self, sample_state_random):
        """Vérifie la cohérence sur plusieurs pas de temps."""
        u0 = sample_state_random
        dt = 0.001
        dx = 0.01
        n_steps = 10
        
        # Legacy
        u_legacy = u0.copy()
        for _ in range(n_steps):
            u_legacy = integrate_ssp_rk3_gpu(
                u_legacy, dt, dx,
                compute_flux_divergence_func=zero_flux_divergence,
                use_fused_kernel=False
            )
        
        # Fused
        u_fused = u0.copy()
        for _ in range(n_steps):
            u_fused = integrate_ssp_rk3_gpu(
                u_fused, dt, dx,
                use_fused_kernel=True
            )
        
        # Vérifier que les résultats sont identiques après plusieurs pas
        np.testing.assert_allclose(
            u_legacy, u_fused,
            atol=1e-11, rtol=1e-11,
            err_msg="Fused kernel diverges from legacy over multiple timesteps"
        )
    
    def test_class_interface_consistency(self, sample_state_random):
        """Vérifie que l'interface de classe fonctionne correctement."""
        u0 = sample_state_random
        N, num_vars = u0.shape
        dt = 0.001
        dx = 0.01
        
        # Test avec la classe directement
        u_n_device = cuda.to_device(u0)
        u_np1_device = cuda.device_array_like(u_n_device)
        
        # Fused
        integrator_fused = SSP_RK3_GPU(N, num_vars, dx, use_fused_kernel=True)
        integrator_fused.integrate_step(u_n_device, u_np1_device, dt)
        u_fused = u_np1_device.copy_to_host()
        
        # Legacy
        u_n_device = cuda.to_device(u0)
        u_np1_device = cuda.device_array_like(u_n_device)
        integrator_legacy = SSP_RK3_GPU(N, num_vars, dx, use_fused_kernel=False)
        integrator_legacy.integrate_step(u_n_device, u_np1_device, dt, zero_flux_divergence)
        u_legacy = u_np1_device.copy_to_host()
        
        np.testing.assert_allclose(u_legacy, u_fused, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    # Exécution simple pour tests rapides
    print("Running SSP-RK3 Fusion Tests...")
    pytest.main([__file__, "-v", "-s"])
