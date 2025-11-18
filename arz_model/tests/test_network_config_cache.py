"""Tests for network configuration caching system."""
import pytest
from pathlib import Path
import json
import tempfile
import shutil

from arz_model.config.network_config_cache import NetworkConfigCache
from arz_model.config.config_factory import CityNetworkConfigFactory, create_victoria_island_config


def test_cache_save_and_load():
    """Test saving and loading config from cache."""
    # Use temporary directory for cache
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = NetworkConfigCache(cache_dir=Path(tmpdir))
        
        # Create a simple config
        csv_path = Path(__file__).parent.parent / 'data' / 'fichier_de_travail_corridor_utf8.csv'
        
        if not csv_path.exists():
            pytest.skip(f"Test data not found: {csv_path}")
        
        factory = CityNetworkConfigFactory(
            city_name="Test City",
            csv_path=str(csv_path),
            use_cache=False  # Don't use auto-cache for this test
        )
        
        config = factory.create_config()
        
        # Compute fingerprint
        fingerprint = cache.compute_fingerprint(
            csv_path=csv_path,
            factory_params=factory.get_params()
        )
        
        # Save to cache
        cache.save(
            config=config,
            city_name="Test City",
            fingerprint=fingerprint,
            csv_path=csv_path,
            factory_params=factory.get_params()
        )
        
        # Load from cache
        loaded_config = cache.load("Test City", fingerprint)
        
        assert loaded_config is not None
        assert len(loaded_config.segments) == len(config.segments)
        assert len(loaded_config.nodes) == len(config.nodes)
        assert loaded_config.time.t_final == config.time.t_final


def test_fingerprint_stability():
    """Test that fingerprint is stable for same inputs."""
    csv_path = Path(__file__).parent.parent / 'data' / 'fichier_de_travail_corridor_utf8.csv'
    
    if not csv_path.exists():
        pytest.skip(f"Test data not found: {csv_path}")
    
    cache = NetworkConfigCache()
    params = {
        'default_density': 20.0,
        'default_velocity': 50.0,
        'cells_per_100m': 4
    }
    
    fingerprint1 = cache.compute_fingerprint(csv_path, None, params)
    fingerprint2 = cache.compute_fingerprint(csv_path, None, params)
    
    assert fingerprint1 == fingerprint2


def test_fingerprint_changes_on_param_change():
    """Test that fingerprint changes when parameters change."""
    csv_path = Path(__file__).parent.parent / 'data' / 'fichier_de_travail_corridor_utf8.csv'
    
    if not csv_path.exists():
        pytest.skip(f"Test data not found: {csv_path}")
    
    cache = NetworkConfigCache()
    
    params1 = {'default_density': 20.0}
    params2 = {'default_density': 25.0}  # Different value
    
    fingerprint1 = cache.compute_fingerprint(csv_path, None, params1)
    fingerprint2 = cache.compute_fingerprint(csv_path, None, params2)
    
    assert fingerprint1 != fingerprint2


def test_cache_clear():
    """Test clearing cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = NetworkConfigCache(cache_dir=Path(tmpdir))
        
        # Create dummy cache files
        city_dir = Path(tmpdir) / "test_city"
        city_dir.mkdir(parents=True)
        
        dummy_file1 = city_dir / "fingerprint1.json"
        dummy_file2 = city_dir / "fingerprint2.json"
        
        dummy_file1.write_text('{"test": "data1"}')
        dummy_file2.write_text('{"test": "data2"}')
        
        # Clear specific city
        deleted = cache.clear("test city")
        assert deleted == 2
        assert not city_dir.exists()


def test_victoria_island_with_cache():
    """Test Victoria Island config creation with cache."""
    csv_path = Path(__file__).parent.parent / 'data' / 'fichier_de_travail_corridor_utf8.csv'
    
    if not csv_path.exists():
        pytest.skip(f"Test data not found: {csv_path}")
    
    # First call - cache miss
    config1 = create_victoria_island_config(csv_path=str(csv_path))
    
    # Second call - cache hit (should be much faster)
    config2 = create_victoria_island_config(csv_path=str(csv_path))
    
    assert len(config1.segments) == len(config2.segments)
    assert len(config1.nodes) == len(config2.nodes)


def test_osm_integration():
    """Test OSM signalized nodes integration."""
    csv_path = Path(__file__).parent.parent / 'data' / 'fichier_de_travail_corridor_utf8.csv'
    enriched_path = Path(__file__).parent.parent / 'data' / 'fichier_de_travail_complet_enriched.xlsx'
    
    if not csv_path.exists():
        pytest.skip(f"Test data not found: {csv_path}")
    
    if not enriched_path.exists():
        pytest.skip(f"Enriched data not found: {enriched_path}")
    
    factory = CityNetworkConfigFactory(
        city_name="Victoria Island",
        csv_path=str(csv_path),
        enriched_path=str(enriched_path),
        use_cache=False
    )
    
    config = factory.create_config()
    
    # Check for signalized nodes
    signalized_nodes = [node for node in config.nodes if node.type == "signalized"]
    
    assert len(signalized_nodes) > 0, "Should detect signalized nodes from OSM data"
    
    # Check traffic light config
    for node in signalized_nodes:
        assert node.traffic_light_config is not None
        assert 'cycle_time' in node.traffic_light_config
        assert 'green_time' in node.traffic_light_config
        assert node.traffic_light_config['cycle_time'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
