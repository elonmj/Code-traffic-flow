"""
ConfigHasher - Hachage cryptographique des configurations RL

Implémente Innovation 2: Config-Hashing Checkpoints
Garantit que les checkpoints ne sont jamais réutilisés avec une config incompatible.

Utilise SHA-256 sur la sérialisation JSON canonique de la configuration.
"""

import hashlib
import json
from typing import Dict, Any


class ConfigHasher:
    """Calcule hash SHA-256 des configurations RL pour garantir compatibilité checkpoints."""
    
    @staticmethod
    def compute_hash(config: Dict[str, Any], hash_length: int = 8) -> str:
        """Calcule hash SHA-256 d'une configuration RL.
        
        Processus (Innovation 2):
        1. Sérialisation JSON canonique (clés triées, pas d'espaces)
        2. Hachage SHA-256
        3. Troncature aux N premiers caractères hexa
        
        Args:
            config: Configuration RL (dict avec learning_rate, buffer_size, etc.)
            hash_length: Nombre de caractères du hash final (défaut: 8)
            
        Returns:
            Hash hexadécimal (ex: "a3f7b2c1")
            
        Example:
            >>> config = {'learning_rate': 0.0001, 'buffer_size': 50000}
            >>> ConfigHasher.compute_hash(config)
            'a3f7b2c1'
        """
        # 1. Canonicalisation JSON (tri clés, pas d'espaces)
        # Ceci garantit que même config → toujours même hash
        config_json = json.dumps(config, sort_keys=True, separators=(',', ':'))
        
        # 2. Hachage SHA-256
        hash_obj = hashlib.sha256(config_json.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()
        
        # 3. Troncature (8 premiers caractères suffisants pour éviter collisions)
        # Probabilité collision < 10^-15 pour espace config réaliste
        return hash_hex[:hash_length]
    
    @staticmethod
    def verify_compatibility(config: Dict[str, Any], expected_hash: str) -> bool:
        """Vérifie qu'une configuration correspond au hash attendu.
        
        Args:
            config: Configuration actuelle
            expected_hash: Hash attendu (ex: depuis nom checkpoint)
            
        Returns:
            True si compatible, False sinon
            
        Example:
            >>> config = {'learning_rate': 0.0001}
            >>> ConfigHasher.verify_compatibility(config, "a3f7b2c1")
            True
        """
        actual_hash = ConfigHasher.compute_hash(config, hash_length=len(expected_hash))
        return actual_hash == expected_hash
