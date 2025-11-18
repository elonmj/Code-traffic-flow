"""
Network Configuration Cache System

Provides persistent caching for NetworkSimulationConfig objects using pickle storage
with hash-based fingerprinting for cache invalidation.
"""
import json
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from .network_simulation_config import NetworkSimulationConfig


class NetworkConfigCache:
    """
    Cache manager for NetworkSimulationConfig objects.
    
    Uses JSON-based persistent storage with MD5 fingerprinting for automatic
    cache invalidation when topology or parameters change.
    
    Cache structure:
        arz_model/cache/{city_name}/{fingerprint}.json
    
    Example:
        >>> cache = NetworkConfigCache()
        >>> fingerprint = cache.compute_fingerprint(csv_path, enriched_path, params)
        >>> config = cache.load("Victoria Island", fingerprint)
        >>> if config is None:
        >>>     config = generate_config()
        >>>     cache.save(config, "Victoria Island", fingerprint, csv_path, params)
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Root cache directory. Defaults to arz_model/cache/
        """
        if cache_dir is None:
            # Default: arz_model/cache/
            self.cache_dir = Path(__file__).parent.parent / "cache"
        else:
            self.cache_dir = Path(cache_dir)
    
    def compute_fingerprint(
        self,
        csv_path: Path,
        enriched_path: Optional[Path] = None,
        factory_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Compute unique fingerprint for network configuration.
        
        Fingerprint based on:
        - CSV file content (hash of file bytes)
        - Enriched data file content (if provided)
        - Factory parameters (sorted JSON)
        
        Args:
            csv_path: Path to topology CSV file
            enriched_path: Optional path to OSM-enriched data
            factory_params: Dict of factory parameters (density, velocity, etc.)
        
        Returns:
            16-character hex fingerprint
        """
        hash_parts = []
        
        # Hash CSV content
        if csv_path.exists():
            with open(csv_path, 'rb') as f:
                csv_hash = hashlib.md5(f.read()).hexdigest()
                hash_parts.append(csv_hash)
        
        # Hash enriched file content if provided
        if enriched_path and enriched_path.exists():
            with open(enriched_path, 'rb') as f:
                enriched_hash = hashlib.md5(f.read()).hexdigest()
                hash_parts.append(enriched_hash)
        
        # Hash factory parameters
        if factory_params:
            params_json = json.dumps(factory_params, sort_keys=True)
            params_hash = hashlib.md5(params_json.encode()).hexdigest()
            hash_parts.append(params_hash)
        
        # Combine all hashes
        combined = ''.join(hash_parts)
        fingerprint = hashlib.md5(combined.encode()).hexdigest()[:16]
        
        return fingerprint
    
    def get_cache_path(self, city_name: str, fingerprint: str) -> Path:
        """
        Get cache file path for a city and fingerprint.
        
        Args:
            city_name: City name (e.g., "Victoria Island")
            fingerprint: 16-char hex fingerprint
        
        Returns:
            Path to cache file
        """
        # Sanitize city name for filesystem
        city_slug = city_name.lower().replace(' ', '_').replace('-', '_')
        city_slug = ''.join(c for c in city_slug if c.isalnum() or c == '_')
        
        cache_file = self.cache_dir / city_slug / f"{fingerprint}.pkl"  # Using pickle now
        return cache_file
    
    def load(self, city_name: str, fingerprint: str) -> Optional[NetworkSimulationConfig]:
        """
        Load cached configuration if it exists.
        
        Args:
            city_name: City name
            fingerprint: Configuration fingerprint
        
        Returns:
            NetworkSimulationConfig if cache hit, None if cache miss
        """
        cache_path = self.get_cache_path(city_name, fingerprint)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Extract config from cache data
            config = cache_data['config']
            
            # Print cache hit info
            created_at = cache_data['metadata'].get('created_at', 'unknown')
            print(f"   ✅ Cache HIT: Loaded config from {fingerprint}.pkl", flush=True)
            print(f"      Cached at: {created_at}", flush=True)
            
            return config
        
        except (pickle.PickleError, KeyError, EOFError) as e:
            print(f"   ⚠️  Cache corrupted for {fingerprint}, will regenerate: {e}", flush=True)
            return None
    
    def save(
        self,
        config: NetworkSimulationConfig,
        city_name: str,
        fingerprint: str,
        csv_path: Path,
        enriched_path: Optional[Path] = None,
        factory_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save configuration to cache.
        
        Args:
            config: NetworkSimulationConfig to cache
            city_name: City name
            fingerprint: Configuration fingerprint
            csv_path: Path to topology CSV (for metadata)
            enriched_path: Optional path to enriched data
            factory_params: Factory parameters used
        """
        cache_path = self.get_cache_path(city_name, fingerprint)
        
        # Create cache directory if needed
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare cache data with metadata
        cache_data = {
            'metadata': {
                'created_at': datetime.utcnow().isoformat() + 'Z',
                'city_name': city_name,
                'fingerprint': fingerprint,
                'csv_path': str(csv_path),
                'enriched_path': str(enriched_path) if enriched_path else None,
                'factory_params': factory_params or {}
            },
            'config': config  # Store the actual Pydantic object directly
        }
        
        # Write to file using pickle
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"   💾 Config cached to: {cache_path.relative_to(self.cache_dir.parent)}", flush=True)
    
    def clear(self, city_name: Optional[str] = None) -> int:
        """
        Clear cache files.
        
        Args:
            city_name: If provided, clear only this city's cache.
                      If None, clear entire cache.
        
        Returns:
            Number of cache files deleted
        """
        if city_name:
            # Clear specific city
            city_slug = city_name.lower().replace(' ', '_').replace('-', '_')
            city_slug = ''.join(c for c in city_slug if c.isalnum() or c == '_')
            city_dir = self.cache_dir / city_slug
            
            if not city_dir.exists():
                return 0
            
            count = 0
            for cache_file in city_dir.glob('*.pkl'):  # Updated for pickle
                cache_file.unlink()
                count += 1
            
            # Remove directory if empty
            if not any(city_dir.iterdir()):
                city_dir.rmdir()
            
            return count
        else:
            # Clear entire cache
            if not self.cache_dir.exists():
                return 0
            
            count = 0
            for cache_file in self.cache_dir.rglob('*.pkl'):  # Updated for pickle
                cache_file.unlink()
                count += 1
            
            # Remove empty directories
            for city_dir in self.cache_dir.iterdir():
                if city_dir.is_dir() and not any(city_dir.iterdir()):
                    city_dir.rmdir()
            
            return count
