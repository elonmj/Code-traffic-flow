# NetworkGrid Flux Solver Integration - Implementation Guide

**Context**: Bug #36 debugging session discovered NetworkGrid lacks junction flux solver  
**Impact**: Traffic signals don't physically block flow (only behavioral coupling works)  
**Priority**: HIGH - Blocks congestion formation test and realistic traffic signal simulation

---

## 🎯 Problem Statement

NetworkGrid currently has:
- ✅ Traffic light infrastructure (TrafficLightController)
- ✅ Behavioral coupling via θ_k (Link class)
- ✅ Boundary conditions for segment endpoints
- ❌ NO junction flux solver (physical flow reduction)

**Result**: RED signals affect driver behavior (θ_k=0) but don't reduce flux passing through junction.

---

## 🔍 Evidence

### What Works:
```python
# Traffic lights correctly report state
has_green_light('seg_1') → False during RED phase
green_segments → [] during RED

# Behavioral coupling correctly calculated  
theta_k → 0.0 during RED (no behavioral memory transfer)
theta_k → 0.5-0.8 during GREEN (normal coupling)

# red_light_factor configured
params.red_light_factor → 0.01 (1% flow during RED)
```

### What Doesn't Work:
```python
# Flux passes through junction unimpeded
densities: 0.079 → 0.072 → 0.065 (draining instead of accumulating)
queue_length: 0.00 veh (should be > 5.0 after 60-120s RED)

# Search confirms no flux solver
grep "solve.*intersection|_calculate_outgoing_flux" network_grid.py
# → NO MATCHES
```

**Root Cause**: `_resolve_node_coupling()` only applies θ_k behavioral coupling, NOT physical flux reduction.

---

## 🏗️ Architecture Analysis

### Current NetworkGrid.step() Flow:
```python
def step(self, dt):
    """Advance simulation by dt seconds."""
    
    # 1. Update each segment independently (ARZ dynamics)
    for segment in self.segments.values():
        segment.step(dt)
    
    # 2. Apply boundary conditions (inflow/outflow)
    self._apply_network_boundary_conditions(self.current_time)
    
    # 3. Apply behavioral coupling ONLY (θ_k)
    self._resolve_node_coupling(self.current_time)
    
    self.current_time += dt
```

**Gap**: No step between boundary conditions and behavioral coupling to:
1. Calculate flux at junctions using Riemann solver
2. Reduce flux based on traffic light state (red_light_factor)
3. Update segment boundaries with reduced flux

---

## 🎯 Required Implementation

### New Method: `_resolve_junction_fluxes()`

**Location**: `arz_model/network/network_grid.py`  
**Insert**: After `_apply_network_boundary_conditions()`, before `_resolve_node_coupling()`

```python
def _resolve_junction_fluxes(self, current_time):
    """
    Calculate and reduce junction fluxes based on traffic signal states.
    
    This method applies physical flux reduction at junctions with traffic lights,
    ensuring RED signals block flow according to red_light_factor parameter.
    
    Steps:
    1. For each junction node with traffic lights
    2. Calculate flux from upstream to downstream segments (Riemann solver)
    3. Apply light_factor: 1.0 if GREEN, red_light_factor if RED
    4. Update downstream segment ghost cells with reduced flux
    
    Note: This is SEPARATE from behavioral coupling (_resolve_node_coupling):
    - Flux reduction: Physical flow capacity at junction
    - θ_k coupling: Driver behavioral response to downstream conditions
    """
    for node_id, node in self.nodes.items():
        # Skip nodes without traffic lights (free-flow junctions)
        if node.traffic_lights is None:
            continue
        
        # Get segments connected to this junction
        incoming_segments = [
            seg for seg in self.segments.values() 
            if seg.end_node == node_id
        ]
        outgoing_segments = [
            seg for seg in self.segments.values() 
            if seg.start_node == node_id
        ]
        
        # Process each outgoing segment
        for out_seg in outgoing_segments:
            # Check traffic light state for this segment
            has_green = node.traffic_lights.has_green_light(out_seg.segment_id)
            
            # Calculate flux reduction factor
            # GREEN (has_green=True): light_factor = 1.0 (100% flow)
            # RED (has_green=False): light_factor = 0.01 (1% flow)
            light_factor = 1.0 if has_green else self.params.red_light_factor
            
            # Calculate junction flux (need to implement or call node_solver)
            # This should use Riemann solver to get flux from upstream state
            flux = self._calculate_junction_flux(
                incoming_segments, 
                out_seg, 
                node, 
                current_time
            )
            
            # Apply traffic light reduction
            reduced_flux = flux * light_factor
            
            # Update downstream segment boundary with reduced flux
            # This modifies ghost cells [0:2] of downstream segment
            self._apply_flux_to_boundary(out_seg, reduced_flux)
```

### Helper Method 1: `_calculate_junction_flux()`

```python
def _calculate_junction_flux(self, incoming_segments, outgoing_segment, node, current_time):
    """
    Calculate flux at junction using Riemann solver.
    
    Args:
        incoming_segments: List of segments ending at junction
        outgoing_segment: Segment starting at junction
        node: Junction node
        current_time: Current simulation time
    
    Returns:
        flux: Array of flux values [vehicle_class, state_vars]
              Shape: (num_vehicle_classes, 4) for ARZ variables
    
    Note: This should integrate with node_solver._calculate_outgoing_flux()
          or implement similar Riemann solver logic.
    """
    # Option 1: Call existing node_solver method
    from arz_model.core.node_solver import _calculate_outgoing_flux
    
    # Get upstream state from incoming segment boundaries
    # Get downstream state from outgoing segment boundaries
    # Call Riemann solver to get flux
    
    # Option 2: Implement simplified junction solver
    # For now, could use simple average or max flux approach
    
    # TODO: Proper implementation needed
    pass
```

### Helper Method 2: `_apply_flux_to_boundary()`

```python
def _apply_flux_to_boundary(self, segment, flux):
    """
    Apply calculated flux to segment boundary (ghost cells).
    
    Args:
        segment: Downstream segment receiving flux
        flux: Flux values to apply (from junction calculation)
    
    Note: Updates ghost cells [0:2] of segment to enforce flux from upstream.
    """
    # Get segment state array
    U = segment.get_state()
    
    # Apply flux to left ghost cells (incoming boundary)
    # This enforces the junction flux as boundary condition
    # U[vehicle_class, 0:2] = ... (convert flux to state)
    
    # TODO: Proper flux-to-state conversion needed
    pass
```

### Updated step() Method:

```python
def step(self, dt):
    """Advance simulation by dt seconds."""
    
    # 1. Update each segment independently (ARZ dynamics)
    for segment in self.segments.values():
        segment.step(dt)
    
    # 2. Apply boundary conditions (inflow/outflow at network edges)
    self._apply_network_boundary_conditions(self.current_time)
    
    # 3. ✨ NEW: Resolve junction fluxes (physical flow reduction)
    self._resolve_junction_fluxes(self.current_time)
    
    # 4. Apply behavioral coupling (θ_k driver response)
    self._resolve_node_coupling(self.current_time)
    
    self.current_time += dt
```

---

## 🧪 Testing Strategy

### Test 1: Flux Reduction Verification
```python
# Setup: 2-segment network with traffic light at junction
# Action: Set RED phase (action=0)
# Expected: Flux through junction < 1% of demand

# Debug output to add:
print(f"[FLUX_DEBUG] Junction flux before light_factor: {flux}")
print(f"[FLUX_DEBUG] light_factor: {light_factor}")
print(f"[FLUX_DEBUG] Junction flux after light_factor: {reduced_flux}")
```

### Test 2: Congestion Formation
```python
# Run existing test:
pytest test_network_integration_quick.py::test_congestion_formation -v

# Expected after fix:
# - queue_length > 5.0 veh after 60-120s RED
# - Densities INCREASE: 0.079 → 0.085 → 0.092
# - Test passes: ✅ PASS: Congestion Formation
```

### Test 3: Congestion Release
```python
# Setup: Form congestion with RED phase
# Action: Switch to GREEN phase (action=1)
# Expected: Queue dissipates, densities decrease
```

---

## 🔗 Integration Points

### With Existing Code:

1. **node_solver.py**: Reuse `_calculate_outgoing_flux()` logic
   - Already has Riemann solver implementation
   - Already calculates flux with traffic light awareness
   - Need to adapt for NetworkGrid segment structure

2. **link.py**: Coordinate with `apply_coupling()`
   - Flux reduction happens BEFORE behavioral coupling
   - θ_k uses post-flux-reduction states for coupling
   - No changes needed in Link class

3. **parameters.py**: Use existing `red_light_factor` parameter
   - Already configured in ModelParameters
   - Already set to 0.01 in builder config
   - No changes needed

4. **traffic_lights.py**: Use existing `has_green_light()` method
   - Already implemented and tested
   - Returns correct values for all phases
   - No changes needed

---

## ⚠️ Implementation Challenges

### Challenge 1: Riemann Solver Complexity
**Problem**: Junction flux calculation non-trivial (shock waves, rarefactions)  
**Solution**: Reuse node_solver._calculate_outgoing_flux() or implement simplified solver  
**Fallback**: Start with simple average flux, refine later

### Challenge 2: Multi-Vehicle Class Handling
**Problem**: Flux array shape (num_classes, 4) needs proper indexing  
**Solution**: Loop over vehicle classes, apply light_factor to each  
**Note**: Current test uses 2 vehicle classes (car, moto)

### Challenge 3: State-Flux Conversion
**Problem**: Flux (flow rate) must be converted to state (density, velocity)  
**Solution**: Use ARZ Lagrangian variables: w = v + p, where p = K * ρ^γ  
**Reference**: Thesis Section 2.2 for ARZ flux-to-state relations

### Challenge 4: Ghost Cell Update Timing
**Problem**: Must update ghost cells without overwriting BC from `_apply_network_boundary_conditions()`  
**Solution**: Only modify internal junction boundaries, not external network boundaries  
**Check**: segment.is_boundary() to distinguish

---

## 📊 Success Metrics

### Immediate (After Implementation):
- [ ] Code compiles without errors
- [ ] No crashes during test execution
- [ ] Debug output shows reduced flux during RED
- [ ] Debug output shows light_factor applied correctly

### Functional (Test Validation):
- [ ] queue_length > 5.0 veh after 60-120s RED signal
- [ ] Upstream densities INCREASE during RED (0.08 → 0.12+)
- [ ] Downstream densities REMAIN LOW during RED (< 0.05)
- [ ] test_congestion_formation passes (✅ PASS)

### Integration (System Behavior):
- [ ] Congestion forms during RED phase
- [ ] Congestion releases during GREEN phase
- [ ] RL training shows learning (loss ↓, reward ↑)
- [ ] Microscope logs show queue dynamics

---

## 🚀 Implementation Steps (Recommended Order)

1. **Create Method Stub** (30 min)
   - Add `_resolve_junction_fluxes()` to network_grid.py
   - Add debug prints to verify it's called
   - Call from step() method
   - Run test to verify no crashes

2. **Implement Light Factor Logic** (1 hour)
   - Loop over nodes with traffic lights
   - Get has_green_light() state
   - Calculate light_factor
   - Add debug print to show light_factor values

3. **Add Flux Calculation** (2-3 hours)
   - Research node_solver._calculate_outgoing_flux()
   - Adapt for NetworkGrid segment structure
   - OR implement simplified Riemann solver
   - Add debug print to show flux before/after reduction

4. **Implement Boundary Update** (1-2 hours)
   - Convert reduced flux to state variables
   - Update downstream segment ghost cells
   - Verify with debug output
   - Check densities increase during RED

5. **Test and Refine** (1-2 hours)
   - Run congestion formation test
   - Analyze queue_length evolution
   - Adjust red_light_factor if needed
   - Remove debug prints after verification

**Total Estimated Time**: 5-9 hours (depending on Riemann solver complexity)

---

## 📖 References

**Code Files**:
- `arz_model/network/network_grid.py` - Implementation location
- `arz_model/core/node_solver.py` - Flux calculation reference
- `arz_model/network/link.py` - Coupling coordination
- `test_network_integration_quick.py` - Test to pass

**Documentation**:
- `BUG_36_TRAFFIC_SIGNAL_COUPLING_FIX_COMPLETE.md` - Bug chain analysis
- `BUG_31_*.md` - Boundary conditions (prerequisite)
- Thesis Section 2.2 - ARZ model equations
- Garavello & Piccoli (2005) - Junction coupling theory

**Debug Session**:
- All configuration bugs fixed (6/7)
- Flux solver is final remaining piece
- Test infrastructure ready for validation

---

## ✅ Pre-Implementation Checklist

Before starting implementation, verify:
- [x] All configuration bugs fixed (Bugs 1-7 from Bug #36 session)
- [x] Traffic lights correctly configured (segment IDs, phases)
- [x] theta_k parameters working (no None returns)
- [x] red_light_factor configured (0.01 in builders.py)
- [x] Test uses correct action (action=0 for RED)
- [x] Debug infrastructure in place (can add flux debug prints)

**Status**: ✅ Ready for implementation

---

*This guide provides complete context and concrete steps to implement junction flux solver in NetworkGrid. Follow the recommended order for systematic progress.*
