# BUG #36 Documentation Index

**Problem**: Traffic signals not blocking flow in NetworkGrid integration test

**Status**: ⚠️ PARTIALLY RESOLVED (6/7 bugs fixed)

---

## 📚 Document Collection

### Primary Documentation

1. **BUG_36_TRAFFIC_SIGNAL_COUPLING_FIX_COMPLETE.md** ⭐ MAIN REPORT
   - Comprehensive fix session report
   - All 7 bugs detailed with before/after code
   - Test result comparisons
   - Technical learnings and architecture insights
   - **Use this for**: Complete understanding of the entire bug chain

2. **BUG_TRAFFIC_SIGNAL_DEBUG_SESSION_COMPLETE.md** ⭐ ALTERNATIVE COMPREHENSIVE
   - Same content as above, different organization
   - Focuses more on diagnostic process
   - Implementation plan for remaining work
   - **Use this for**: Step-by-step debugging perspective

### Quick Reference

3. **BUG_36_DOCUMENTATION_INDEX.md** 📍 THIS FILE
   - Navigation hub for all Bug #36 docs
   - Quick overview of available resources

### Note on Other Files

The following files exist in workspace but are for DIFFERENT bugs:
- `BUG_36_ANALYSIS_CHECKLIST.md` - About Bug #36 GPU validation (different issue)
- `BUG_36_FIX_SUMMARY.md` - About Bug #36 GPU inflow BC (different issue)

**Important**: Bug numbers got reused across different problem domains. Always check document content to ensure correct context.

---

## 🎯 Bug Summary (Quick Reference)

**Found**: 7 interconnected bugs  
**Fixed**: 6 bugs (86% complete)  
**Remaining**: 1 architecture gap (flux solver)

### Fixed Bugs:
1. ✅ Phase config used geographic names instead of segment IDs
2. ✅ TrafficLightConfig not transferred through runner
3. ✅ TrafficLightController not created from config
4. ✅ Phase order inverted (GREEN=0, RED=1)
5. ✅ theta_* parameters returned None (no defaults)
6. ✅ red_light_factor too high (10% → 1%)
7. ✅ Test used wrong action (1 → 0 for RED)

### Remaining Work:
- ❌ NetworkGrid lacks junction flux solver (architecture enhancement needed)

---

## 🔍 Finding Information

**Want to understand the bug chain?**
→ Read `BUG_36_TRAFFIC_SIGNAL_COUPLING_FIX_COMPLETE.md` sections "Bugs Found and Fixed"

**Want to see code changes?**
→ Read `BUG_36_TRAFFIC_SIGNAL_COUPLING_FIX_COMPLETE.md` section "Code Changes Summary"

**Want to implement the fix?**
→ Read `BUG_TRAFFIC_SIGNAL_DEBUG_SESSION_COMPLETE.md` section "Implementation Plan"

**Want test verification steps?**
→ Read either main doc section "Test Results"

**Want diagnostic commands?**
→ Both docs include verification commands and debug output analysis

---

## 🚀 Next Action for Future Work

**Implement Junction Flux Solver**:
1. Location: `arz_model/network/network_grid.py`
2. Method: Add `_resolve_junction_fluxes()` before `_resolve_node_coupling()`
3. Logic: Calculate flux with Riemann solver + apply red_light_factor
4. Test: Run `pytest test_network_integration_quick.py::test_congestion_formation`
5. Success: queue_length > 5.0 veh after 60-120s

**Estimated Time**: 4-6 hours (implementation + testing)

---

## 📖 Related Bug Documentation

**Prerequisites**:
- `BUG_31_*.md` - Boundary conditions (must be working for congestion test)

**Architecture**:
- `ARCHITECTURE_UXSIM_INTEGRATION_PLAN.md` - Network design
- `ARZ_MODEL_PACKAGE_ARCHITECTURAL_AUDIT_COMPLETE.md` - Package structure

---

## ⏱️ Session Metrics

**Debug Duration**: ~4-5 hours  
**Bugs Discovered**: 7  
**Bugs Fixed**: 6  
**Test Status**: Still failing (but for known reason)  
**ROI**: High (configuration infrastructure now robust)

---

*Use this index to navigate Bug #36 documentation efficiently. All configuration bugs are fixed; only flux solver integration remains.*
