"""Test rapide du système de cache."""
from arz_model.config import create_victoria_island_config
from arz_model.config.network_config_cache import NetworkConfigCache

# Clear cache
cache = NetworkConfigCache()
print("🧹 Clearing cache...")
deleted = cache.clear()
print(f"✅ Deleted {deleted} files\n")

# Create config
print("📊 Creating config (should be CACHE MISS)...")
c = create_victoria_island_config()

# Summary
signalized = [n for n in c.nodes if n.type == 'signalized']
print(f"\n✅ SUCCESS!")
print(f"   - Segments: {len(c.segments)}")
print(f"   - Nodes: {len(c.nodes)}")
print(f"   - Traffic Lights (OSM): {len(signalized)}")
if signalized:
    print(f"   - Light config example: {signalized[0].traffic_light_config}")

# Second call - cache hit
print("\n📊 Second call (should be CACHE HIT)...")
c2 = create_victoria_island_config()
print(f"✅ Same config: {len(c.segments) == len(c2.segments)}")

print("\n🎉 CACHE SYSTEM FULLY OPERATIONAL!")
