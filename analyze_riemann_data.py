import numpy as np
import os

INPUT_DIR = r"d:\Projets\Alibi\Code project\kaggle\results\generic-test-runner-kernel\thesis_stage1"

def analyze_file(filename, test_name):
    filepath = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"MISSING: {filename}")
        return

    data = np.load(filepath, allow_pickle=True)
    U = data['U']
    x = data['x']
    
    # U shape is (4, N) -> [rho_m, v_m, rho_c, v_c]
    rho_m = U[0, :]
    v_m = U[1, :]
    rho_c = U[2, :]
    v_c = U[3, :]
    
    print(f"\n--- ANALYZING: {test_name} ({filename}) ---")
    print(f"Grid size: {len(x)}")
    
    # Check for negativity
    min_rho = np.min(U[[0, 2], :])
    print(f"Min Density: {min_rho:.6f} (Should be >= 0)")
    
    # Analyze gradients to distinguish shock vs rarefaction
    # We look at the active class (the one with significant density change)
    
    # Determine active class based on max density range
    range_m = np.max(rho_m) - np.min(rho_m)
    range_c = np.max(rho_c) - np.min(rho_c)
    
    if range_m > range_c:
        active_rho = rho_m
        active_name = "Motos"
    else:
        active_rho = rho_c
        active_name = "Cars"
        
    print(f"Active Class: {active_name}")
    
    # Calculate gradient
    grad = np.gradient(active_rho)
    max_grad = np.max(np.abs(grad))
    print(f"Max Gradient: {max_grad:.6f}")
    
    # Check for oscillations (Total Variation)
    tv = np.sum(np.abs(np.diff(active_rho)))
    total_change = np.abs(active_rho[-1] - active_rho[0])
    print(f"Total Variation: {tv:.6f}")
    print(f"Net Change: {total_change:.6f}")
    print(f"Ratio TV/Change: {tv/total_change if total_change > 1e-6 else 0:.2f} (Should be ~1 for monotonic)")

    # Specific checks
    if "choc" in filename:
        # Shock should have high gradient and be monotonic
        if max_grad > 0.01: # Arbitrary threshold for "sharp"
            print("Feature: SHARP TRANSITION DETECTED (Consistent with Shock)")
        else:
            print("Feature: SMOOTH TRANSITION (Inconsistent with Shock?)")
            
    if "detente" in filename:
        # Rarefaction should be smooth
        if max_grad < 0.01: # Arbitrary threshold
            print("Feature: SMOOTH TRANSITION DETECTED (Consistent with Rarefaction)")
        else:
            print("Feature: SHARP TRANSITION (Inconsistent with Rarefaction?)")

    if "vide" in filename:
        print(f"Min Density in Vide test: {min_rho:.6e}")
        if min_rho > -1e-9:
             print("Feature: POSITIVITY PRESERVED")
        else:
             print("Feature: NEGATIVITY DETECTED")

analyze_file('riemann_choc_simple_motos.npz', 'Test 1: Choc Simple')
analyze_file('riemann_detente_voitures.npz', 'Test 2: Détente')
analyze_file('riemann_apparition_vide_motos.npz', 'Test 3: Apparition Vide')
analyze_file('riemann_discontinuite_contact.npz', 'Test 4: Contact')
analyze_file('riemann_interaction_multiclasse.npz', 'Test 5: Interaction')
