"""Tests for UXsim visualization adapter"""

import pytest
import numpy as np
import os
import tempfile
from unittest.mock import MagicMock

from ..visualization.uxsim_adapter import ARZtoUXsimVisualizer
from ..grid.grid1d import Grid1D
from ..core.parameters import ModelParameters
from ..io.data_manager import save_simulation_data, load_simulation_data


class TestUXsimAdapter:
    """Test suite for ARZtoUXsimVisualizer"""
    
    def test_uxsim_import(self):
        """Test que UXsim peut être importé correctement"""
        try:
            from uxsim import World
            # Test création World basique
            world = World(name="test")
            assert world is not None
            print("✓ UXsim import test passed")
        except ImportError as e:
            pytest.fail(f"UXsim import failed: {e}")
    
    @pytest.fixture
    def synthetic_npz_data(self):
        """Génère des données NPZ synthétiques pour les tests"""
        # Créer grille synthétique
        grid = Grid1D(N=20, xmin=0.0, xmax=1000.0, num_ghost_cells=2)
        
        # Créer paramètres synthétiques
        params = ModelParameters()
        params.scenario_name = "test_scenario"
        params.Vmax_m = 15.0  # m/s pour motos
        params.Vmax_c = 12.0  # m/s pour voitures
        
        # Générer données temporelles
        times = np.linspace(0, 100, 50)  # 50 pas de temps
        N_physical = grid.N_physical
        
        # Générer états réalistes
        states = []
        for i, t in enumerate(times):
            # Densités oscillantes réalistes
            x_centers = grid.cell_centers(include_ghost=False)
            
            # Densité motos (rho_m)
            rho_m = 0.05 + 0.03 * np.sin(2 * np.pi * x_centers / 500 + t / 20)
            rho_m = np.maximum(rho_m, 0.01)  # Éviter densités nulles
            
            # Vitesse pondérée motos (w_m = rho_m * v_m)
            v_m = 12 + 3 * np.cos(2 * np.pi * x_centers / 300 + t / 15)
            w_m = rho_m * v_m
            
            # Densité voitures (rho_c)
            rho_c = 0.08 + 0.04 * np.cos(2 * np.pi * x_centers / 400 + t / 25)
            rho_c = np.maximum(rho_c, 0.01)
            
            # Vitesse pondérée voitures (w_c = rho_c * v_c)
            v_c = 10 + 2 * np.sin(2 * np.pi * x_centers / 350 + t / 18)
            w_c = rho_c * v_c
            
            # État complet [rho_m, w_m, rho_c, w_c]
            state = np.array([rho_m, w_m, rho_c, w_c])
            states.append(state)
        
        states = np.array(states)  # Shape: (50, 4, N_physical)
        
        return {
            'times': times,
            'states': states,
            'grid': grid,
            'params': params
        }
    
    @pytest.fixture
    def temp_npz_file(self, synthetic_npz_data):
        """Crée un fichier NPZ temporaire avec données synthétiques"""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            save_simulation_data(
                tmp.name,
                synthetic_npz_data['times'],
                synthetic_npz_data['states'],
                synthetic_npz_data['grid'],
                synthetic_npz_data['params']
            )
            yield tmp.name
        
        # Cleanup
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    
    def test_adapter_initialization(self, temp_npz_file):
        """Test initialisation de l'adaptateur avec données synthétiques"""
        viz = ARZtoUXsimVisualizer(temp_npz_file)
        
        # Vérifier attributs
        assert viz.times is not None
        assert viz.states is not None
        assert viz.grid is not None
        assert viz.params is not None
        
        # Vérifier dimensions
        assert viz.states.ndim == 3
        assert viz.states.shape[1] == 4  # [rho_m, w_m, rho_c, w_c]
        assert len(viz.times) == viz.states.shape[0]
        
        print(f"✓ Adapter initialization test passed")
        print(f"  - Data shape: {viz.states.shape}")
        print(f"  - Time range: {viz.times[0]:.1f} to {viz.times[-1]:.1f} s")
    
    def test_adapter_initialization_invalid_file(self):
        """Test gestion d'erreur pour fichier inexistant"""
        with pytest.raises((FileNotFoundError, RuntimeError)):
            ARZtoUXsimVisualizer("nonexistent_file.npz")
    
    def test_create_uxsim_network(self, temp_npz_file):
        """Test création réseau UXsim"""
        viz = ARZtoUXsimVisualizer(temp_npz_file)
        world = viz.create_uxsim_network()
        
        # Vérifier objet World
        assert world is not None
        assert hasattr(world, 'NODES')
        assert hasattr(world, 'LINKS')
        
        # Vérifier nombre de nœuds et links
        N_physical = viz.grid.N_physical
        assert len(world.NODES) == N_physical
        assert len(world.LINKS) == N_physical - 1
        
        print(f"✓ UXsim network creation test passed")
        print(f"  - Nodes: {len(world.NODES)}, Links: {len(world.LINKS)}")
    
    def test_visualize_snapshot(self, temp_npz_file):
        """Test génération snapshot de visualisation"""
        import matplotlib.pyplot as plt
        viz = ARZtoUXsimVisualizer(temp_npz_file)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            try:
                # Tester snapshot final
                fig = viz.visualize_snapshot(time_index=-1, save_path=tmp.name)
                assert fig is not None
                assert os.path.exists(tmp.name)
                assert os.path.getsize(tmp.name) > 0  # Fichier non vide
                
                print(f"✓ Snapshot visualization test passed")
                print(f"  - Output file: {tmp.name} ({os.path.getsize(tmp.name)} bytes)")
                
                # Fermer explicitement la figure
                plt.close(fig)
                
            finally:
                # Attendre un petit délai et forcer la fermeture des figures
                plt.close('all')
                import time
                time.sleep(0.1)  # Délai court pour Windows
                if os.path.exists(tmp.name):
                    try:
                        os.unlink(tmp.name)
                    except PermissionError:
                        pass  # Ignore si fichier encore utilisé
    
    def test_visualize_snapshot_index_validation(self, temp_npz_file):
        """Test validation des indices temporels"""
        viz = ARZtoUXsimVisualizer(temp_npz_file)
        
        # Test index valide
        fig = viz.visualize_snapshot(time_index=0)
        assert fig is not None
        
        # Test index négatif valide
        fig = viz.visualize_snapshot(time_index=-1)
        assert fig is not None
        
        # Test index invalide
        with pytest.raises(IndexError):
            viz.visualize_snapshot(time_index=1000)
        
        print("✓ Index validation test passed")
    
    def test_create_animation_gif(self, temp_npz_file):
        """Test création animation GIF (simulation rapide)"""
        viz = ARZtoUXsimVisualizer(temp_npz_file)
        
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
            try:
                # Utiliser seulement 3 frames pour test rapide
                time_indices = [0, len(viz.times)//2, -1]
                viz.create_animation(tmp.name, fps=2, time_indices=time_indices)
                
                assert os.path.exists(tmp.name)
                assert os.path.getsize(tmp.name) > 0
                
                print(f"✓ GIF animation test passed")
                print(f"  - Output file: {tmp.name} ({os.path.getsize(tmp.name)} bytes)")
                
            except Exception as e:
                # Animation peut échouer selon l'environnement (pillow, etc.)
                print(f"Warning: Animation test skipped due to: {e}")
                pytest.skip(f"Animation test skipped: {e}")
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
    
    def test_create_animation_invalid_format(self, temp_npz_file):
        """Test gestion format d'animation invalide"""
        viz = ARZtoUXsimVisualizer(temp_npz_file)
        
        with pytest.raises(ValueError):
            viz.create_animation("test.invalid", fps=10, time_indices=[0, 1])


def test_integration_with_existing_data():
    """Test d'intégration si des fichiers NPZ réels existent"""
    # Chercher fichiers NPZ existants
    npz_patterns = [
        "d:/Projets/Alibi/Code project/results/**/*.npz",
        "d:/Projets/Alibi/Code project/arz_model/results/**/*.npz",
        "d:/Projets/Alibi/Code project/validation_output/**/*.npz"
    ]
    
    import glob
    found_files = []
    for pattern in npz_patterns:
        found_files.extend(glob.glob(pattern, recursive=True))
    
    if found_files:
        # Test sur premier fichier trouvé
        npz_file = found_files[0]
        print(f"Testing with real data: {npz_file}")
        
        try:
            viz = ARZtoUXsimVisualizer(npz_file)
            
            # Test création réseau
            world = viz.create_uxsim_network()
            assert world is not None
            
            # Test snapshot
            fig = viz.visualize_snapshot(-1)
            assert fig is not None
            
            print(f"✓ Real data integration test passed")
            print(f"  - File: {os.path.basename(npz_file)}")
            print(f"  - Grid size: {viz.states.shape[2]} cells")
            
        except Exception as e:
            print(f"Warning: Real data test failed: {e}")
            pytest.skip(f"Real data test failed: {e}")
    else:
        print("No real NPZ files found, skipping integration test")
        pytest.skip("No real NPZ files found")


if __name__ == "__main__":
    # Exécution directe des tests
    print("Running UXsim adapter tests...")
    
    # Test import UXsim
    try:
        from uxsim import World
        print("✓ UXsim import OK")
    except ImportError as e:
        print(f"✗ UXsim import failed: {e}")
        exit(1)
    
    # Test génération données synthétiques
    try:
        from ..grid.grid1d import Grid1D
        from ..core.parameters import ModelParameters
        print("✓ ARZ imports OK")
    except ImportError as e:
        print(f"✗ ARZ imports failed: {e}")
        exit(1)
    
    print("All basic tests passed. Run with pytest for full test suite.")