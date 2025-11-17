"""
Script to verify the contents of the network_simulation_results.pkl file.
"""
import os
import pickle
import numpy as np

def main():
    """Main verification function."""
    print("=" * 70)
    print("=== Verifying network_simulation_results.pkl ===")
    print("=" * 70)

    # Path to the results file from the last Kaggle run
    results_path = os.path.join(
        'kaggle', 'results', 'elonmj_generic-test-runner-kernel', 'simulation_results', 'network_simulation_results.pkl'
    )

    if not os.path.exists(results_path):
        print(f"❌ ERROR: Results file not found at '{results_path}'")
        return

    print(f"✅ Found results file: {results_path}")

    try:
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        print("✅ Successfully loaded the pickle file.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load pickle file: {e}")
        return

    if not isinstance(results, dict):
        print(f"❌ ERROR: Expected results to be a dictionary, but got {type(results)}")
        return

    print(f"✅ Results object is a dictionary with keys: {list(results.keys())}")
    
    all_valid = True
    for seg_id, data in results.items():
        print(f"\n--- Verifying Segment: {seg_id} ---")
        if 'U_history' not in data:
            print(f"❌ ERROR: 'U_history' not found for segment {seg_id}")
            all_valid = False
            continue
        
        U_history = data['U_history']
        if not isinstance(U_history, np.ndarray):
            print(f"❌ ERROR: 'U_history' for {seg_id} is not a numpy array, but {type(U_history)}")
            all_valid = False
            continue
            
        print(f"  - U_history shape: {U_history.shape}")

        # Check the final state
        final_U = U_history[-1]
        
        # Check if all values are zero
        if np.all(final_U == 0):
            print("❌ VALIDATION FAILED: The final state vector is all zeros.")
            all_valid = False
        else:
            print("✅ VALIDATION PASSED: The final state vector contains non-zero values.")

        # Print some stats for manual inspection
        stats = {
            'rho_m': {'min': np.min(final_U[0, :]), 'max': np.max(final_U[0, :]), 'mean': np.mean(final_U[0, :])},
            'w_m':   {'min': np.min(final_U[1, :]), 'max': np.max(final_U[1, :]), 'mean': np.mean(final_U[1, :])},
            'rho_c': {'min': np.min(final_U[2, :]), 'max': np.max(final_U[2, :]), 'mean': np.mean(final_U[2, :])},
            'w_c':   {'min': np.min(final_U[3, :]), 'max': np.max(final_U[3, :]), 'mean': np.mean(final_U[3, :])},
        }
        
        print("  - Final State Statistics:")
        for var, s in stats.items():
            print(f"    - {var}: Min={s['min']:.2f}, Max={s['max']:.2f}, Mean={s['mean']:.2f}")

    print("\n" + "=" * 70)
    if all_valid:
        print("✅✅✅ OVERALL VERIFICATION SUCCEEDED ✅✅✅")
    else:
        print("❌❌❌ OVERALL VERIFICATION FAILED ❌❌❌")
    print("=" * 70)

if __name__ == "__main__":
    main()
