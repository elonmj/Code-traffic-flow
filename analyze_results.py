import pickle
import numpy as np
import matplotlib.pyplot as plt

def analyze_results(filepath):
    """
    Loads and analyzes the simulation results from a pickle file.

    Args:
        filepath (str): The path to the .pkl file.
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

    # --- 1. Inspect Data Structure ---
    print("\n--- Data Structure ---")
    if isinstance(data, dict):
        print("File contains a dictionary with the following keys:")
        for key, value in data.items():
            value_type = type(value)
            value_shape = ""
            if hasattr(value, 'shape'):
                value_shape = f" (shape: {value.shape})"
            elif isinstance(value, list):
                value_shape = f" (length: {len(value)})"
            elif isinstance(value, dict):
                value_shape = f" (keys: {list(value.keys())})"
            print(f"  - '{key}': {value_type}{value_shape}")
    else:
        print(f"File contains an object of type: {type(data)}")
        # Add more specific checks if needed for other data types
        return

    # --- 2. Basic Simulation Analysis ---
    print("\n--- Simulation Analysis ---")
    if 'history' in data and isinstance(data['history'], dict):
        history_data = data['history']
        if 'time' in history_data and 'segments' in history_data:
            time_history = history_data['time']
            segment_history_dict = history_data['segments']

            print(f"  - Simulation ended at t = {time_history[-1]:.2f} seconds.")
            print(f"  - Total time steps recorded: {len(time_history)}")

            # --- 3. Detailed Segment Analysis & Visualization ---
            num_segments = len(segment_history_dict)
            print(f"  - Number of road segments: {num_segments}")

            # Create a directory for plots
            import os
            output_dir = "results_analysis"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print(f"  - Saving plots to '{output_dir}/'")

            # Analyze and plot for each segment
            for seg_id, segment_steps in segment_history_dict.items():
                try:
                    # Extract density for the first vehicle type (e.g., cars)
                    # Shape of each item in segment_steps: (num_vars, num_cells)
                    density_car_history = [step[0, :] for step in segment_steps]

                    # Plot Spatiotemporal Diagram for Density
                    plt.figure(figsize=(12, 8))
                    # Ensure we have a 2D array to plot
                    if np.array(density_car_history).ndim == 1:
                        density_car_history = np.array(density_car_history)[np.newaxis, :]
                    
                    plt.imshow(np.array(density_car_history), aspect='auto', cmap='viridis', origin='lower',
                               extent=[0, density_car_history[0].shape[0], 0, time_history[-1]])
                    plt.colorbar(label='Car Density (veh/m)')
                    plt.xlabel('Cell Index')
                    plt.ylabel('Time (s)')
                    plt.title(f'Spatiotemporal Diagram of Car Density - {seg_id}')
                    plot_path = os.path.join(output_dir, f'spatiotemporal_density_{seg_id}.png')
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"    - Saved density plot for {seg_id}: {plot_path}")

                except IndexError as e:
                    print(f"    - ⚠️ Could not analyze segment {seg_id}. Data might be incomplete. Error: {e}")
                except Exception as e:
                    print(f"    - ❌ An unexpected error occurred during analysis of segment {seg_id}: {e}")
        else:
            print("  - Could not find 'time' or 'segments' keys within the 'history' dictionary.")

    else:
        print("  - Could not find 'history' key for analysis.")
if __name__ == "__main__":
    # The results file is expected to be in the same directory as the script
    # after being moved from the kaggle artifacts.
    analyze_results('network_simulation_results.pkl')
