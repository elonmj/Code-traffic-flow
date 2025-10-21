# Architecture Extension Strategy for NetworkGrid Implementation

**Date**: 2025-01-21  
**Context**: SUMO/CityFlow architectural analysis completed  
**Goal**: Determine how to extend `arz_model` for multi-segment network support

---

## 📋 CURRENT ARCHITECTURE ANALYSIS

### Complete Module Structure

```
arz_model/
│
├── 📁 core/                    # Physics & Core Entities
│   ├── parameters.py           # ARZ model parameters (τ, ν, etc.)
│   ├── physics.py              # Fundamental density, equilibrium velocity
│   ├── node_solver.py          # Riemann solver at junctions (EXISTS!)
│   ├── intersection.py         # Intersection management
│   └── traffic_lights.py       # Traffic signal control
│
├── 📁 grid/                    # Computational Domains
│   ├── grid1d.py               # Single-segment grid (CURRENT)
│   └── __init__.py
│
├── 📁 numerics/                # Numerical Methods
│   ├── riemann_solvers.py      # ARZ, LWR Riemann solvers
│   ├── boundary_conditions.py  # Inflow/outflow BCs
│   ├── time_integration.py     # RK2, RK3 time stepping
│   ├── network_coupling.py     # Multi-segment coupling (INCOMPLETE)
│   ├── cfl.py                  # CFL condition calculator
│   └── reconstruction/         # WENO5, MUSCL schemes
│       ├── weno5.py
│       └── muscl.py
│
├── 📁 simulation/              # High-Level Orchestration
│   ├── runner.py               # Main simulation loop
│   ├── initial_conditions.py   # IC generators (riemann, stationary)
│   └── __init__.py
│
├── 📁 io/                      # Input/Output
│   ├── data_manager.py         # Save/load results (.npz files)
│   └── __init__.py
│
├── 📁 visualization/           # Plotting & Visualization
│   ├── plotting.py             # Matplotlib plots
│   ├── uxsim_adapter.py        # UXSim integration
│   └── __init__.py
│
├── 📁 calibration/             # Digital Twin Calibration System
│   ├── core/
│   │   ├── calibration_runner.py
│   │   ├── network_builder.py    # ⚠️ User is currently here!
│   │   ├── parameter_set.py
│   │   └── parameters.py
│   ├── data/
│   │   ├── real_data_loader.py
│   │   ├── tomtom_collector.py
│   │   └── group_manager.py
│   ├── metrics/
│   │   ├── calibration_metrics.py
│   │   └── validation.py
│   └── optimizers/
│       ├── bayesian_optimizer.py
│       └── grid_search.py
│
├── 📁 analysis/                # Post-Simulation Analysis
│   ├── metrics.py              # TVD, L1 error, etc.
│   ├── conservation.py         # Mass conservation checks
│   └── convergence.py          # Convergence analysis
│
└── 📁 tests/                   # Test Suite
    ├── test_network_system.py  # Network system tests (EXISTS)
    ├── test_physics.py
    └── test_simulation_runner_rl.py
```

---

## 🔍 WHAT'S MISSING FOR MULTI-SEGMENT NETWORKS

### Current State (Single Segment)

```python
# arz_model/grid/grid1d.py
class Grid1D:
    """Single road segment computational domain"""
    def __init__(self, xmin: float, xmax: float, N: int):
        self.xmin = xmin  # One segment start
        self.xmax = xmax  # One segment end
        self.N = N        # Cells in this segment
        self.dx = (xmax - xmin) / N
        
        # State vectors for ONE segment
        self.rho = np.zeros(N + 2 * nghost)  # Density
        self.q = np.zeros(N + 2 * nghost)    # Flow
```

**Problem**: 
- ❌ Can only represent ONE road segment
- ❌ No concept of multiple segments
- ❌ No node connectivity
- ❌ No network topology

### What Professional Simulators Have

#### SUMO Architecture
```cpp
MSNet                          // Network coordinator
  ├─ MSEdgeControl            // Container for ALL edges
  │   └─ vector<MSEdge*>      // All segments in network
  │       └─ MSEdge           // One segment
  │           ├─ vector<MSLane*>
  │           ├─ MSJunction* myFromJunction
  │           └─ MSJunction* myToJunction
  └─ MSJunctionControl        // Container for ALL junctions
      └─ vector<MSJunction*>
```

#### CityFlow Architecture
```cpp
RoadNet                        // Network coordinator
  ├─ vector<Road>             // All segments
  │   └─ Road                 // One segment
  │       ├─ vector<Lane>
  │       ├─ Intersection* startIntersection
  │       └─ Intersection* endIntersection
  ├─ vector<Intersection>     // All nodes
  └─ vector<Lane*>            // Flat list for efficiency
```

---

## 🎯 THREE ARCHITECTURAL OPTIONS

### Option 1: Extend Existing Folders ⚠️

**Structure**:
```
arz_model/
├── core/
│   ├── node_solver.py         # EXISTS
│   ├── node.py                # NEW - Node entity
│   └── link.py                # NEW - Link entity
├── grid/
│   ├── grid1d.py → segment_grid.py  # RENAME
│   └── network_grid.py        # NEW - Network coordinator
├── numerics/
│   └── network_coupling.py    # REFACTOR (remove TODOs)
```

**Pros**:
- ✅ Minimal file changes
- ✅ Follows existing module boundaries
- ✅ `core/` for entities, `grid/` for domains

**Cons**:
- ❌ `grid/` now has TWO abstractions (segment vs network)
- ❌ Confusing: "Is grid the segment or the network?"
- ❌ No clear place for future graph algorithms

**Verdict**: ⚠️ **NOT RECOMMENDED** - Mixes segment and network concerns

---

### Option 2: New `network/` Module ✅ RECOMMENDED

**Structure**:
```
arz_model/
├── network/                   # ✨ NEW MODULE
│   ├── __init__.py
│   ├── network_grid.py        # Top-level coordinator (like MSNet)
│   ├── node.py                # Node class (wraps NodeSolver)
│   ├── link.py                # Link class (segment coupling)
│   └── topology.py            # Graph utilities (future: shortest path)
│
├── grid/
│   ├── __init__.py
│   └── segment_grid.py        # RENAMED from grid1d.py
│
├── core/
│   ├── node_solver.py         # KEEP - node physics
│   └── ... (existing)
│
├── numerics/
│   ├── network_coupling.py    # REFACTOR to use network/
│   └── ... (existing)
```

**Separation of Concerns**:
- `network/` = **Topology** (graph structure, connectivity)
- `grid/` = **Computational Domain** (discretization, cells)
- `core/` = **Physics** (equations, parameters)
- `numerics/` = **Numerical Methods** (solvers, time stepping)

**Pros**:
- ✅ **Clear module boundary** for network concepts
- ✅ **Self-documenting**: `from arz_model.network import NetworkGrid`
- ✅ **Matches industry pattern** (SUMO has `netload/`, CityFlow has `roadnet/`)
- ✅ **Scalable**: Room for routing, traffic assignment, graph algorithms
- ✅ **Minimal breaking changes**: Only rename grid1d → segment_grid

**Cons**:
- ⚠️ New top-level module (one more folder)
- ⚠️ Requires updating imports across codebase

**Migration Strategy**:
```python
# grid/__init__.py - Backwards compatibility
from .segment_grid import SegmentGrid
from .segment_grid import SegmentGrid as Grid1D  # Alias for old code

import warnings
def __getattr__(name):
    if name == 'Grid1D':
        warnings.warn(
            "Grid1D is deprecated, use SegmentGrid instead",
            DeprecationWarning,
            stacklevel=2
        )
        return SegmentGrid
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

**Verdict**: ✅ **STRONGLY RECOMMENDED**

---

### Option 3: Hierarchical `grid/` Module 📦

**Structure**:
```
arz_model/
├── grid/
│   ├── __init__.py
│   ├── base/
│   │   ├── computational_domain.py  # Abstract base class
│   │   └── grid_interface.py
│   ├── segment/
│   │   └── segment_grid.py
│   ├── network/
│   │   ├── network_grid.py
│   │   ├── node.py
│   │   └── link.py
│   └── utilities/
│       └── mesh_generators.py
```

**Pros**:
- ✅ Everything grid-related in one place
- ✅ Clear hierarchy with submodules

**Cons**:
- ❌ Deep nesting: `grid/network/network_grid.py`
- ❌ Import paths: `from arz_model.grid.network.network_grid import NetworkGrid`
- ❌ Over-engineered for current scale
- ❌ "Network" is not a "grid" conceptually

**Verdict**: ❌ **NOT RECOMMENDED** - Over-complicated

---

## 🏆 FINAL RECOMMENDATION: Option 2 (New `network/` Module)

### Why This Is The Best Choice

1. **Matches Professional Patterns**:
   - SUMO: `microsim/network/` separate from `microsim/edges/`
   - CityFlow: `roadnet/` as dedicated module
   - Industry-proven separation of concerns

2. **Clear Semantic Meaning**:
   - `network` = Graph topology, connectivity, routing
   - `grid` = Computational domain, discretization
   - No confusion about what each module does

3. **Scalability**:
   - Future features naturally fit: routing algorithms, traffic assignment, OD matrices
   - Won't clutter existing modules
   - Clear place to add graph utilities

4. **Minimal Disruption**:
   - Backwards-compatible Grid1D alias
   - Gradual migration path
   - Most code unchanged initially

5. **User is Already Thinking This Way**:
   - `calibration/core/network_builder.py` exists!
   - Already building network graphs from CSV
   - Natural to have `network/` module to consume this

---

## 📐 DETAILED IMPLEMENTATION PLAN

### Phase 1: Create `network/` Module Structure

```bash
mkdir arz_model/network
touch arz_model/network/__init__.py
touch arz_model/network/network_grid.py
touch arz_model/network/node.py
touch arz_model/network/link.py
```

### Phase 2: Implement Core Classes

#### `network/node.py`
```python
"""
Network Node
============

Represents a node in the traffic network (intersection or boundary).
Wraps the NodeSolver for physics computations.
"""

from typing import List, Optional, Dict
from ..core.node_solver import NodeSolver
from ..core.parameters import Parameters

class Node:
    """
    A node in the traffic network.
    
    Connects multiple road segments and resolves traffic flow
    at junctions using Riemann solvers.
    """
    
    def __init__(
        self,
        node_id: str,
        position: float = 0.0,
        is_boundary: bool = False
    ):
        """
        Args:
            node_id: Unique identifier
            position: Spatial position (for 1D networks, can be abstract)
            is_boundary: True if this is a network boundary (inflow/outflow)
        """
        self.node_id = node_id
        self.position = position
        self.is_boundary = is_boundary
        
        # Connected segments (will be set by NetworkGrid)
        self.incoming_segments: List[str] = []
        self.outgoing_segments: List[str] = []
        
        # Physics solver
        self.solver: Optional[NodeSolver] = None
        
    def initialize_solver(self, params: Parameters):
        """Initialize the Riemann solver at this node"""
        if not self.is_boundary:
            self.solver = NodeSolver(params)
            
    def add_incoming_segment(self, segment_id: str):
        """Register an incoming road segment"""
        if segment_id not in self.incoming_segments:
            self.incoming_segments.append(segment_id)
            
    def add_outgoing_segment(self, segment_id: str):
        """Register an outgoing road segment"""
        if segment_id not in self.outgoing_segments:
            self.outgoing_segments.append(segment_id)
            
    @property
    def is_intersection(self) -> bool:
        """True if this node connects multiple segments"""
        return len(self.incoming_segments) + len(self.outgoing_segments) > 2
        
    def __repr__(self) -> str:
        return (
            f"Node(id={self.node_id}, "
            f"in={len(self.incoming_segments)}, "
            f"out={len(self.outgoing_segments)}, "
            f"boundary={self.is_boundary})"
        )
```

#### `network/link.py`
```python
"""
Network Link
============

Represents a connection between road segments at a node.
Handles coupling logic, priorities, and traffic signal control.
"""

from typing import Optional
from enum import Enum

class LinkPriority(Enum):
    """Priority levels for link conflicts"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    
class TrafficSignalState(Enum):
    """Traffic signal states"""
    GREEN = 0
    YELLOW = 1
    RED = 2

class Link:
    """
    A link connecting two segments at a node.
    
    Inspired by SUMO's MSLink and CityFlow's LaneLink.
    Manages coupling logic and traffic control.
    """
    
    def __init__(
        self,
        link_id: str,
        from_segment: str,
        to_segment: str,
        via_node: str,
        priority: LinkPriority = LinkPriority.NONE
    ):
        """
        Args:
            link_id: Unique identifier
            from_segment: Source segment ID
            to_segment: Destination segment ID
            via_node: Node connecting them
            priority: Link priority for conflict resolution
        """
        self.link_id = link_id
        self.from_segment = from_segment
        self.to_segment = to_segment
        self.via_node = via_node
        self.priority = priority
        
        # Traffic signal state
        self.signal_state: Optional[TrafficSignalState] = None
        self.has_signal = False
        
        # Turn type (for future routing)
        self.turn_type: Optional[str] = None  # 'left', 'right', 'straight'
        
    def set_traffic_signal(self, state: TrafficSignalState):
        """Set traffic signal state"""
        self.has_signal = True
        self.signal_state = state
        
    def is_passable(self) -> bool:
        """Check if vehicles can pass through this link"""
        if self.has_signal:
            return self.signal_state == TrafficSignalState.GREEN
        return True  # No signal = always passable
        
    def __repr__(self) -> str:
        signal_str = f", signal={self.signal_state.name}" if self.has_signal else ""
        return (
            f"Link({self.from_segment} → {self.to_segment} "
            f"via {self.via_node}{signal_str})"
        )
```

#### `network/network_grid.py`
```python
"""
Network Grid
============

Top-level coordinator for multi-segment traffic networks.
Inspired by SUMO's MSNet and CityFlow's RoadNet.
"""

from typing import Dict, List, Optional
import numpy as np

from ..grid.segment_grid import SegmentGrid
from .node import Node
from .link import Link
from ..core.parameters import Parameters

class NetworkGrid:
    """
    Top-level network coordinator.
    
    Manages multiple road segments, nodes, and their connections.
    Coordinates simulation across the entire network.
    """
    
    def __init__(self, params: Parameters):
        """
        Args:
            params: Global ARZ parameters
        """
        self.params = params
        
        # Collections (SUMO pattern: centralized containers)
        self.segments: Dict[str, SegmentGrid] = {}
        self.nodes: Dict[str, Node] = {}
        self.links: List[Link] = []
        
        # Simulation state
        self.time = 0.0
        self.dt = 0.0  # Will be computed from CFL
        
    def add_segment(
        self,
        segment_id: str,
        xmin: float,
        xmax: float,
        N: int,
        start_node: str,
        end_node: str
    ) -> SegmentGrid:
        """
        Add a road segment to the network.
        
        Args:
            segment_id: Unique identifier
            xmin, xmax: Segment spatial bounds
            N: Number of cells
            start_node, end_node: Node IDs at segment ends
            
        Returns:
            Created SegmentGrid object
        """
        # Create segment
        segment = SegmentGrid(xmin, xmax, N, self.params)
        segment.segment_id = segment_id
        segment.start_node = start_node
        segment.end_node = end_node
        
        self.segments[segment_id] = segment
        
        # Register connections at nodes
        if start_node not in self.nodes:
            self.nodes[start_node] = Node(start_node)
        self.nodes[start_node].add_outgoing_segment(segment_id)
        
        if end_node not in self.nodes:
            self.nodes[end_node] = Node(end_node)
        self.nodes[end_node].add_incoming_segment(segment_id)
        
        return segment
        
    def add_link(
        self,
        from_segment: str,
        to_segment: str,
        via_node: str
    ) -> Link:
        """
        Add a link connecting two segments.
        
        Args:
            from_segment: Source segment ID
            to_segment: Destination segment ID
            via_node: Node connecting them
            
        Returns:
            Created Link object
        """
        link_id = f"{from_segment}_to_{to_segment}"
        link = Link(link_id, from_segment, to_segment, via_node)
        self.links.append(link)
        return link
        
    def initialize(self):
        """Initialize all segments and nodes"""
        # Initialize node solvers
        for node in self.nodes.values():
            node.initialize_solver(self.params)
            
        # Initialize segment grids
        for segment in self.segments.values():
            segment.initialize()
            
    def step(self, dt: float):
        """
        Advance simulation by one time step.
        
        This is where the multi-segment coupling happens:
        1. Update each segment independently
        2. Resolve fluxes at nodes using NodeSolver
        3. Apply coupling boundary conditions
        
        Args:
            dt: Time step size
        """
        self.dt = dt
        
        # Step 1: Update segments (independent evolution)
        for segment in self.segments.values():
            segment.step(dt)
            
        # Step 2: Resolve node coupling (Riemann problems)
        self._resolve_node_coupling()
        
        # Step 3: Apply coupling BCs to segments
        self._apply_coupling_bcs()
        
        self.time += dt
        
    def _resolve_node_coupling(self):
        """
        Resolve traffic flow at all nodes using Riemann solvers.
        
        This is the KEY multi-segment coupling logic!
        """
        for node_id, node in self.nodes.items():
            if node.is_boundary or node.solver is None:
                continue
                
            # Get states from incoming/outgoing segments
            incoming_states = []
            outgoing_states = []
            
            for seg_id in node.incoming_segments:
                segment = self.segments[seg_id]
                # Get right boundary state (end of segment)
                rho = segment.rho[-segment.params.nghost]
                q = segment.q[-segment.params.nghost]
                incoming_states.append((rho, q))
                
            for seg_id in node.outgoing_segments:
                segment = self.segments[seg_id]
                # Get left boundary state (start of segment)
                rho = segment.rho[segment.params.nghost]
                q = segment.q[segment.params.nghost]
                outgoing_states.append((rho, q))
                
            # Solve Riemann problem at node
            flux = node.solver.solve(incoming_states, outgoing_states)
            
            # Store flux for BC application
            node.flux = flux
            
    def _apply_coupling_bcs(self):
        """
        Apply node fluxes as boundary conditions to segments.
        """
        for node_id, node in self.nodes.items():
            if not hasattr(node, 'flux'):
                continue
                
            flux = node.flux
            
            # Apply to incoming segments (right boundary)
            for seg_id in node.incoming_segments:
                segment = self.segments[seg_id]
                segment.set_right_boundary_flux(flux)
                
            # Apply to outgoing segments (left boundary)
            for seg_id in node.outgoing_segments:
                segment = self.segments[seg_id]
                segment.set_left_boundary_flux(flux)
                
    def get_total_vehicles(self) -> float:
        """Get total number of vehicles in network"""
        total = 0.0
        for segment in self.segments.values():
            total += np.sum(segment.rho[segment.params.nghost:-segment.params.nghost])
        return total * self.segments[list(self.segments.keys())[0]].dx
        
    def __repr__(self) -> str:
        return (
            f"NetworkGrid(segments={len(self.segments)}, "
            f"nodes={len(self.nodes)}, "
            f"links={len(self.links)}, "
            f"time={self.time:.2f})"
        )
```

### Phase 3: Rename `grid1d.py` → `segment_grid.py`

Just rename the file, add `segment_id`, `start_node`, `end_node` attributes:

```python
class SegmentGrid:
    """
    Computational grid for a single road segment.
    
    Renamed from Grid1D to clarify that this represents ONE segment
    in a potentially multi-segment network.
    """
    
    def __init__(self, xmin: float, xmax: float, N: int, params: Parameters):
        # Existing code...
        
        # NEW: Network topology info
        self.segment_id: Optional[str] = None
        self.start_node: Optional[str] = None
        self.end_node: Optional[str] = None
```

### Phase 4: Update Imports

```python
# arz_model/network/__init__.py
from .network_grid import NetworkGrid
from .node import Node
from .link import Link, LinkPriority, TrafficSignalState

__all__ = [
    'NetworkGrid',
    'Node',
    'Link',
    'LinkPriority',
    'TrafficSignalState'
]
```

```python
# arz_model/grid/__init__.py
from .segment_grid import SegmentGrid

# Backwards compatibility
from .segment_grid import SegmentGrid as Grid1D

import warnings

def __getattr__(name):
    if name == 'Grid1D':
        warnings.warn(
            "Grid1D is deprecated, use SegmentGrid instead",
            DeprecationWarning,
            stacklevel=2
        )
        return SegmentGrid
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['SegmentGrid', 'Grid1D']
```

---

## 🔗 INTEGRATION WITH EXISTING CODE

### How `calibration/core/network_builder.py` Connects

**Current State** (from attached file):
```python
class NetworkBuilder:
    def __init__(self):
        self.segments: Dict[str, RoadSegment] = {}
        self.nodes: Dict[str, NetworkNode] = {}
        self.intersections: List[Intersection] = []  # TODO: Uncomment
```

**After Extension**:
```python
from arz_model.network import NetworkGrid, Node

class NetworkBuilder:
    """Builds ARZ NetworkGrid from CSV corridor data"""
    
    def build_network_grid(self, corridor_file: str) -> NetworkGrid:
        """
        Build NetworkGrid from corridor CSV.
        
        Returns:
            Ready-to-simulate NetworkGrid object
        """
        # Load corridor data
        df = pd.read_csv(corridor_file)
        
        # Create network
        params = Parameters()  # Or load from config
        network = NetworkGrid(params)
        
        # Add segments
        for _, row in df.iterrows():
            segment_id = f"{row['u']}_{row['v']}"
            
            # Assume 100 cells per segment (or compute from length/dx)
            N = 100
            xmin = 0.0
            xmax = row['length']
            
            network.add_segment(
                segment_id=segment_id,
                xmin=xmin,
                xmax=xmax,
                N=N,
                start_node=str(row['u']),
                end_node=str(row['v'])
            )
            
        # Add links at intersections
        for node_id, node in network.nodes.items():
            if node.is_intersection:
                # Create links between all incoming/outgoing pairs
                for in_seg in node.incoming_segments:
                    for out_seg in node.outgoing_segments:
                        network.add_link(in_seg, out_seg, node_id)
                        
        # Initialize
        network.initialize()
        
        return network
```

**Usage**:
```python
from arz_model.calibration.core.network_builder import NetworkBuilder

# Build network from Victoria Island corridor
builder = NetworkBuilder()
network = builder.build_network_grid("victoria_island_corridor.csv")

# Simulate
for step in range(1000):
    dt = compute_cfl_timestep(network)
    network.step(dt)
    
# Analyze
total_vehicles = network.get_total_vehicles()
print(f"Total vehicles: {total_vehicles}")
```

---

## 📊 COMPARISON: Before vs After

### Before (Single Segment)

```python
from arz_model.grid.grid1d import Grid1D

# Can only simulate ONE segment
grid = Grid1D(xmin=0.0, xmax=1000.0, N=100)
grid.set_initial_condition(...)
grid.simulate(T=3600)
```

**Limitation**: No network support, no junctions, no multi-segment

### After (Multi-Segment Network)

```python
from arz_model.network import NetworkGrid
from arz_model.grid.segment_grid import SegmentGrid  # Still available!

# Option 1: Use NetworkGrid for multi-segment
network = NetworkGrid(params)
network.add_segment("seg1", 0, 1000, 100, "node_A", "node_B")
network.add_segment("seg2", 0, 800, 80, "node_B", "node_C")
network.add_link("seg1", "seg2", "node_B")
network.simulate(T=3600)

# Option 2: Still use SegmentGrid for single segment (backwards compatible!)
segment = SegmentGrid(0, 1000, 100, params)
segment.simulate(T=3600)
```

**Gain**: Multi-segment support WITHOUT breaking existing single-segment code!

---

## 🎯 EFFORT ESTIMATES

### Phase 1: Module Creation (1 day)
- Create `network/` folder
- Implement `Node` class (150 lines)
- Implement `Link` class (100 lines)
- Write unit tests

### Phase 2: NetworkGrid Core (3-4 days)
- Implement `NetworkGrid` class (300 lines)
- Implement `add_segment()`, `add_link()`
- Implement `_resolve_node_coupling()`
- Implement `_apply_coupling_bcs()`
- Write integration tests

### Phase 3: SegmentGrid Refactoring (2 days)
- Rename `grid1d.py` → `segment_grid.py`
- Add network topology attributes
- Add BC methods for coupling
- Update all imports with backwards compatibility
- Test existing code still works

### Phase 4: Integration & Testing (2-3 days)
- Update `network_builder.py` to use `NetworkGrid`
- Create example networks (2-segment, 3-segment, intersection)
- Validation tests (mass conservation, flux continuity)
- Documentation

### Phase 5: RL Environment Update (1-2 days)
- Update RL environment to use `NetworkGrid`
- Test reward function with multi-segment
- Verify training works

**TOTAL: 10-14 days** (2-3 weeks as estimated)

---

## ✅ VALIDATION CHECKLIST

After implementation, verify:

- [ ] `NetworkGrid` can create multi-segment networks
- [ ] `Node` objects correctly connect segments
- [ ] `Link` objects manage coupling
- [ ] Node Riemann solver resolves fluxes correctly
- [ ] Mass conservation across entire network
- [ ] Backwards compatibility: old `Grid1D` code still works
- [ ] `network_builder.py` can build `NetworkGrid` from CSV
- [ ] RL environment works with new architecture
- [ ] Documentation updated
- [ ] All tests pass

---

## 🚀 NEXT STEPS

**IMMEDIATE** (if approved):
1. Create `arz_model/network/` folder
2. Implement `Node`, `Link`, `NetworkGrid` classes
3. Rename `grid1d.py` → `segment_grid.py` with compatibility
4. Write tests
5. Update `network_builder.py` integration

**User, your decision?**
- ✅ Go ahead with Option 2 (new `network/` module)?
- 🤔 Prefer Option 1 (extend existing folders)?
- 💭 Different approach entirely?

Let me know and I'll start implementing immediately! 🎯
