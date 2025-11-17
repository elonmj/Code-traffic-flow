#  Unified Network Visualization System

## Quick Start

Generate all visualizations with one command:

```bash
python generate_visuals.py all
```

## Generated Files

This system creates **3 essential visualizations**:

1. **network_topology.png** - Static network map with highlighted active segments
2. **network_snapshots.png** - Multi-panel traffic state snapshots
3. **network_animation.gif** - Animated traffic flow evolution

## Usage

```bash
# Generate all 3 visualizations
python generate_visuals.py all

# Generate only topology
python generate_visuals.py topology

# Generate only snapshots  
python generate_visuals.py snapshots

# Generate only animation
python generate_visuals.py animation

# Show help
python generate_visuals.py --help
```

## Architecture

The system uses the official **arz_model/visualization/** module:

```
SimulationDataLoader  NetworkTopologyBuilder  NetworkTrafficVisualizer
     (Load Data)          (Build Graph)            (Render Visuals)
```

## Animation Requirements

 **Animation requires time-series data**

If you see an error when generating animation:
1. Open: rz_model/main_full_network_simulation.py
2. Find: TimeConfig section
3. Set: output_dt = 1.0 (save every second)
4. Re-run: python arz_model/main_full_network_simulation.py
5. Retry: python generate_visuals.py animation

## Clean System

 **One unified script** replaces 9+ scattered visualization scripts  
 **3 essential outputs** instead of 30+ redundant files  
 **Intelligent error messages** guide you when data is insufficient  
 **Uses official module** - no workarounds or hacks  

---

*Generated: November 16, 2025*
