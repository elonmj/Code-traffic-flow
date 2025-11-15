import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def robust_analyze_results(filepath):
    """
    Loads and robustly analyzes simulation results from a pickle file,
    handling various data structures and potential inconsistencies.
    """
    print(f"🔬 Analyzing results from: {filepath}")

    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {filepath}")
        return
    except Exception as e:
        print(f"❌ An error occurred while loading the file: {e}")
        return

    # --- 1. Deep Inspect Data Structure ---
    print("\n--- Data Structure Inspection ---")
    if not isinstance(data, dict):
        print(f"File contains a non-dictionary object of type: {type(data)}")
        return

    print("Data Keys:", list(data.keys()))
    history = data.get('history')
    if not isinstance(history, dict):
        print("❌ 'history' key not found or is not a dictionary.")
        return

    print("History Keys:", list(history.keys()))
    time_history = history.get('time')
    segment_history = history.get('segments')

    if time_history is None or segment_history is None:
        print("❌ 'time' or 'segments' not found in history.")
        return

    # --- 2. Basic Simulation Analysis ---
    print("\n--- Simulation Summary ---")
    print(f"  - Simulation ended at t = {time_history[-1]:.4f} seconds.")
    print(f"  - Total time steps recorded: {len(time_history)}")

    if len(time_history) <= 1:
        print("  - ⚠️ Warning: Simulation only contains initial state (1 time step). Spatiotemporal plots will not show evolution.")

    # --- 3. Detailed Segment Analysis & Visualization ---
    print("\n--- Segment Analysis & Visualization ---")
    output_dir = "results_analysis_final"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"  - Saving plots to '{output_dir}/'")

    if not isinstance(segment_history, dict):
        print(f"❌ Expected 'segments' to be a dictionary, but got {type(segment_history)}")
        return
    
    # --- Diagnostic Print ---
    print(f"\n--- Pre-loop Diagnostic ---")
    print(f"Type of segment_history: {type(segment_history)}")
    print(f"Keys in segment_history: {list(segment_history.keys())}")
    print(f"Number of segments found: {len(segment_history)}")
    # --- End Diagnostic Print ---
        
    for seg_id, variables_dict in segment_history.items():
        print(f"\n  Analyzing Segment: '{seg_id}'...")
        try:
            if not isinstance(variables_dict, dict):
                print(f"    - ⚠️ Skipping '{seg_id}': Expected a dictionary of variables, but got {type(variables_dict)}.")
                continue

            # The data is already separated by variable. Get the 'density' list.
            density_states_over_time = variables_dict.get('density')

            if not density_states_over_time or not isinstance(density_states_over_time, list):
                print(f"    - ⚠️ Skipping '{seg_id}': No 'density' data list found.")
                continue
            
            # Convert the list of 1D arrays into a single 2D numpy array.
            density_car_history = np.array(density_states_over_time)

            if density_car_history.ndim == 1:
                density_car_history = density_car_history[np.newaxis, :]

            print(f"    - Data shape for density: {density_car_history.shape} (time_steps, cells)")

            # Plot Spatiotemporal Diagram for Density
            plt.figure(figsize=(12, 8))
            
            # Handle single time step case for extent
            t_end = time_history[-1] if len(time_history) > 1 else 1.0

            plt.imshow(density_car_history, aspect='auto', cmap='viridis', origin='lower',
                       extent=[0, density_car_history.shape[1], 0, t_end])
            
            plt.colorbar(label='Car Density (veh/m)')
            plt.xlabel('Cell Index')
            plt.ylabel('Time (s)')
            plt.title(f'Spatiotemporal Diagram of Car Density - {seg_id}')
            
            plot_path = os.path.join(output_dir, f'spatiotemporal_density_{seg_id}.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"    - ✅ Saved density plot: {plot_path}")

        except Exception as e:
            print(f"    - ❌ An unexpected error occurred during analysis of segment '{seg_id}': {e}")
            import traceback
            traceback.print_exc()
if __name__ == "__main__":
    analyze_results_path = 'network_simulation_results.pkl'
    if not os.path.exists(analyze_results_path):
        print(f"FATAL: Results file not found at '{analyze_results_path}'. Make sure it's in the project root.")
    else:
        robust_analyze_results(analyze_results_path)
