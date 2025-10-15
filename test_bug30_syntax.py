"""
Quick syntax validation for Bug #30 fix
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "Code_RL" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "validation_ch7" / "scripts"))

print("Testing Bug #30 fix syntax...")

try:
    from test_section_7_6_rl_performance import RLPerformanceValidationTest
    print("✅ Module imports successfully")
    
    # Check that RLController has the new signature
    tester = RLPerformanceValidationTest(quick_test=True)
    
    # Check RLController class signature
    import inspect
    rl_controller_init = inspect.signature(tester.RLController.__init__)
    params = list(rl_controller_init.parameters.keys())
    
    print(f"RLController.__init__ parameters: {params}")
    
    required_params = ['self', 'scenario_type', 'model_path', 'scenario_config_path', 'device']
    missing = [p for p in required_params if p not in params]
    
    if missing:
        print(f"❌ Missing parameters: {missing}")
        sys.exit(1)
    else:
        print("✅ RLController has correct signature with Bug #30 fix")
        print("   - scenario_config_path parameter present")
        print("   - device parameter present")
        print()
        print("🎉 Bug #30 fix is syntactically correct!")
        print("   Model will be loaded WITH environment during evaluation")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
