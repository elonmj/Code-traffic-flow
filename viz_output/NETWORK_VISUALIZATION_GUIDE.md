# Lagos Network Topology Visualization Guide

## Overview

This guide explains the three network topology visualizations created from the ARZ two-class traffic flow simulation of Lagos roads.

**Created**: 2025-11-15  
**Simulation**: 30-minute traffic flow on Lagos corridor  
**Network**: 70 road segments, 60 intersections  
**Roads**: Akin Adesola Street, Ahmadu Bello Way, Adeola Odeku Street, Saka Tinubu Street

---

## Files Created

### 1. `network_topology.png` - Static Network View

**Purpose**: Shows the complete road network structure

**What You See**:
- **Nodes** (white circles with black border): Intersections where roads meet
- **Edges** (colored arrows): Road segments connecting intersections
- **Colors by Road Type**:
  - **Blue (thick)**: Primary roads - main arteries (33 segments)
  - **Red (medium)**: Secondary roads - connecting routes (23 segments)
  - **Orange (thin)**: Tertiary roads - local streets (14 segments)

**How to Read It**:
- Arrow directions show traffic flow direction (one-way streets)
- Layout uses force-directed algorithm (spring layout) to minimize edge crossings
- Node positions are optimized for clarity, not geographic accuracy

**Use Cases**:
- Understanding network structure and connectivity
- Identifying critical intersections (high degree nodes)
- Seeing road hierarchy (primary vs secondary vs tertiary)
- Planning network modifications or expansions

---

### 2. `network_animation.gif` - Traffic Flow Evolution

**Purpose**: Animated visualization of traffic speed changes over 30 minutes

**What You See**:
- **Edge Colors by Average Speed**:
  - **Green**: > 60 km/h (free flow conditions)
  - **Orange**: 30-60 km/h (moderate congestion)
  - **Red**: < 30 km/h (heavy congestion)
- **Time Counter**: Shows simulation time in seconds and minutes
- **Speed Legend**: Explains color coding

**Animation Details**:
- **Duration**: 30 minutes of simulation time
- **Frame Rate**: 10 FPS (frames per second)
- **Total Frames**: 180 frames
- **File Format**: GIF (plays in browsers, presentations, documents)

**How to Read It**:
- Watch how congestion propagates through the network
- Identify bottleneck locations (edges that stay red)
- Observe recovery patterns (red → orange → green transitions)
- Notice time-dependent patterns (rush hour effects)

**Use Cases**:
- Presenting simulation results to non-technical audiences
- Identifying congestion hot spots dynamically
- Demonstrating traffic wave propagation
- Comparing before/after scenarios in presentations

---

### 3. `network_snapshots.png` - Key Time Points

**Purpose**: Six snapshot views of the network at critical time moments

**Panel Layout**: 2 rows × 3 columns (6 panels total)

**Time Points**:
1. **t = 0s (0 min)** - Initial conditions
2. **t = 360s (6 min)** - Early development
3. **t = 720s (12 min)** - Mid-simulation state
4. **t = 1080s (18 min)** - Later evolution
5. **t = 1440s (24 min)** - Near-final state
6. **t = 1800s (30 min)** - Final conditions

**What Each Panel Shows**:
- Network topology with same color coding as animation
- **Panel Title**: Time and average network speed
- Edge colors indicate traffic state at that moment

**How to Read It**:
- Compare panels to see temporal evolution
- Track specific edges across time points
- Identify persistent vs transient congestion
- Quantify overall network performance by average speed

**Use Cases**:
- Comparing specific time points side-by-side
- Creating presentation slides (6 snapshots in one figure)
- Analyzing critical moments (e.g., when congestion peaks)
- Documenting simulation outcomes in reports

---

## Comparison with Other Visualizations

### Previous Visualizations Created:
1. **Spatiotemporal Diagrams** (`visualize_results.py`):
   - Individual segment density/speed evolution
   - Time on x-axis, space on y-axis
   - Detailed but segment-specific

2. **Fundamental Diagrams** (`create_fundamental_diagrams.py`):
   - Flow-density relationships
   - Academic standard plots
   - Theory validation

3. **Public Dashboards** (`create_simple_public_visualizations.py`):
   - GPS-style congestion maps
   - Emoji-based infographics
   - General public communication

### Network Topology Visualizations (This Set):
- **Unique Value**: Shows complete network structure and interconnections
- **Perspective**: Bird's-eye view of entire system
- **Audience**: Engineers, planners, researchers
- **Focus**: Network-level patterns and propagation

---

## Technical Details

### Software Stack
- **Python**: Core programming language
- **NetworkX 3.4.2**: Graph construction and layouts
- **Matplotlib**: Rendering and export
- **Pandas 2.2.2**: CSV data loading
- **NumPy**: Numerical computations
- **Pillow**: GIF animation export

### Architecture Pattern
**Separation of Concerns** (Dijkstra, 1974):
- **Concern 1**: Data loading (`data_loader.py`)
- **Concern 2**: Graph building (`network_builder.py`)
- **Concern 3**: Visualization rendering (`network_visualizer.py`)

### Data Sources
- **Topology**: `arz_model/data/fichier_de_travail_corridor_utf8.csv`
- **Simulation Results**: `network_simulation_results.pkl`
- **Layout Algorithm**: Spring layout (force-directed, k=0.5, iterations=100, seed=42)

---

## Interpretation Guide

### Colors and Meanings

#### Static Topology (`network_topology.png`)
| Color | Road Type | Meaning | Count |
|-------|-----------|---------|-------|
| Blue (thick) | Primary | Main arteries, high capacity | 33 |
| Red (medium) | Secondary | Connecting roads, medium capacity | 23 |
| Orange (thin) | Tertiary | Local streets, lower capacity | 14 |

#### Traffic State (`network_animation.gif`, `network_snapshots.png`)
| Color | Speed Range | Traffic Condition | Expected LOS |
|-------|-------------|-------------------|--------------|
| Green | > 60 km/h | Free flow | LOS A-B |
| Orange | 30-60 km/h | Moderate congestion | LOS C-D |
| Red | < 30 km/h | Heavy congestion | LOS E-F |

*LOS = Level of Service (Highway Capacity Manual standard)*

### Common Patterns

1. **Bottleneck Formation**:
   - Red edges upstream of intersections
   - Indicates capacity constraints
   - Look for recurring patterns across time

2. **Congestion Propagation**:
   - Red patches spreading upstream
   - Traffic waves moving against flow direction
   - Shows network-level phenomena

3. **Recovery Dynamics**:
   - Red → Orange → Green transitions
   - Speed indicates traffic dissipation
   - May be uneven across network

---

## Troubleshooting

### Common Issues

**Q: Why are node positions not geographically accurate?**  
A: The CSV file lacks GPS coordinates. We use spring layout algorithm to create a readable visualization optimized for clarity, not geographic accuracy.

**Q: Why does the animation show similar colors for all edges?**  
A: Current implementation uses simplified segment-to-edge mapping. In production, would map simulation segments to specific graph edges by matching node IDs or road names.

**Q: How do I create custom time snapshots?**  
A: Edit the `snapshot_times` list in `create_network_topology_visualizations.py`:
```python
snapshot_times = [0, 300, 600, 900, 1200, 1500]  # Your custom times in seconds
```

**Q: Can I change the color scheme?**  
A: Yes! Modify the speed thresholds in `network_visualizer.py`, `create_traffic_animation()` method:
```python
if avg_speed > 70:      # Your threshold
    color = '#00CC00'   # Your color
```

**Q: How do I regenerate visualizations?**  
A: Simply run:
```bash
python create_network_topology_visualizations.py
```
All files in `viz_output/` will be regenerated.

---

## References

### Traffic Simulation Visualization
- **SUMO**: https://sumo.dlr.de/docs/Tutorials/quick_start.html
- **MATSim**: https://www.matsim.org/gallery

### Network Analysis
- **NetworkX**: https://networkx.org/documentation/stable/
- **Spring Layout**: Force-directed graph drawing (Fruchterman & Reingold, 1991)

### Software Engineering
- **Separation of Concerns**: Dijkstra, E. W. (1974). "On the role of scientific thought"

---

**Last Updated**: 2025-11-15  
**Version**: 1.0  
**Author**: ARZ Traffic Simulation Team
