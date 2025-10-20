"""
Tests unitaires ConfigHasher

Teste Innovation 2: Config-Hashing Checkpoints
- Hashing SHA-256 déterministe
- Détection incompatibilités config
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from domain.checkpoint.config_hasher import ConfigHasher


class TestConfigHasher:
    """Tests unitaires ConfigHasher."""
    
    def test_compute_hash_deterministic(self):
        """Test hash déterministe (même config → même hash)."""
        # Given
        config = {
            "algorithm": "dqn",
            "hyperparameters": {
                "learning_rate": 0.0001,
                "buffer_size": 100000
            }
        }
        
        # When
        hash1 = ConfigHasher.compute_hash(config)
        hash2 = ConfigHasher.compute_hash(config)
        
        # Then
        assert hash1 == hash2
        assert len(hash1) == 8  # Longueur par défaut
    
    def test_compute_hash_different_configs(self):
        """Test hash différents pour configs différentes."""
        # Given
        config1 = {"algorithm": "dqn", "learning_rate": 0.0001}
        config2 = {"algorithm": "ppo", "learning_rate": 0.0003}
        
        # When
        hash1 = ConfigHasher.compute_hash(config1)
        hash2 = ConfigHasher.compute_hash(config2)
        
        # Then
        assert hash1 != hash2
    
    def test_compute_hash_order_independent(self):
        """Test hash identique indépendamment de l'ordre clés."""
        # Given
        config1 = {"b": 2, "a": 1, "c": 3}
        config2 = {"a": 1, "c": 3, "b": 2}
        
        # When
        hash1 = ConfigHasher.compute_hash(config1)
        hash2 = ConfigHasher.compute_hash(config2)
        
        # Then
        assert hash1 == hash2  # Ordre clés ne doit pas affecter hash
    
    def test_compute_hash_custom_length(self):
        """Test hash avec longueur personnalisée."""
        # Given
        config = {"algorithm": "dqn"}
        
        # When
        hash_8 = ConfigHasher.compute_hash(config, hash_length=8)
        hash_16 = ConfigHasher.compute_hash(config, hash_length=16)
        
        # Then
        assert len(hash_8) == 8
        assert len(hash_16) == 16
        assert hash_16.startswith(hash_8)  # Hash plus long commence par hash court
    
    def test_verify_compatibility_success(self):
        """Test vérification compatibilité config - succès."""
        # Given
        config = {
            "algorithm": "dqn",
            "hyperparameters": {"learning_rate": 0.0001}
        }
        expected_hash = ConfigHasher.compute_hash(config)
        
        # When
        is_compatible = ConfigHasher.verify_compatibility(config, expected_hash)
        
        # Then
        assert is_compatible is True
    
    def test_verify_compatibility_failure(self):
        """Test vérification compatibilité config - échec."""
        # Given
        config = {"algorithm": "dqn", "learning_rate": 0.0001}
        wrong_hash = "wronghash"
        
        # When
        is_compatible = ConfigHasher.verify_compatibility(config, wrong_hash)
        
        # Then
        assert is_compatible is False
    
    def test_hash_sensitivity_to_hyperparameters(self):
        """Test sensibilité hash aux changements hyperparamètres."""
        # Given
        config_base = {
            "algorithm": "dqn",
            "hyperparameters": {
                "learning_rate": 0.0001,
                "buffer_size": 100000
            }
        }
        
        # Config avec learning_rate modifié
        config_modified = {
            "algorithm": "dqn",
            "hyperparameters": {
                "learning_rate": 0.0002,  # Changement ici
                "buffer_size": 100000
            }
        }
        
        # When
        hash_base = ConfigHasher.compute_hash(config_base)
        hash_modified = ConfigHasher.compute_hash(config_modified)
        
        # Then
        assert hash_base != hash_modified  # Innovation 2: Détecte incompatibilité
    
    def test_hash_with_nested_config(self):
        """Test hash avec configuration imbriquée complexe."""
        # Given
        complex_config = {
            "algorithm": "ppo",
            "hyperparameters": {
                "learning_rate": 0.0003,
                "network_arch": {
                    "pi": [64, 64],
                    "vf": [64, 64]
                }
            },
            "total_timesteps": 100000
        }
        
        # When
        hash1 = ConfigHasher.compute_hash(complex_config)
        hash2 = ConfigHasher.compute_hash(complex_config)
        
        # Then
        assert hash1 == hash2
        assert len(hash1) == 8


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
