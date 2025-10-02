# code/visualize_results.py
import argparse
import os
import sys
import glob
import numpy as np

# Ensure the project root is in sys.path for absolute imports
# Assuming this script is in code/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from code.io.data_manager import load_simulation_data
    from code.visualization.plotting import plot_profiles, plot_spacetime
    # Import Grid1D and ModelParameters for type hinting if needed,
    # but they are loaded within the data dictionary
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure the script is run correctly (e.g., 'python -m code.visualize_results ...')")
    print("and all required modules (io, visualization) are present.")
    sys.exit(1)

def find_latest_npz(results_dir: str) -> str | None:
    """Finds the most recently modified .npz file in the specified directory."""
    list_of_files = glob.glob(os.path.join(results_dir, '*.npz'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file

def main():
    parser = argparse.ArgumentParser(description="Visualize results from an ARZ simulation.")
    parser.add_argument(
        '-i', '--input',
        help="Path to a specific simulation result (.npz) file. If not provided, --scenario_name must be set to find the latest .npz in results/<scenario_name>."
    )
    parser.add_argument(
        '--scenario_name',
        help="Name of the scenario (used to find latest results in results/<scenario_name>/ if --input is not provided)."
    )
    parser.add_argument(
        '--results_dir',
        default='results',
        help="Directory containing simulation result files (default: results)."
    )
    parser.add_argument(
        '--plots',
        nargs='+',
        default=['all'],
        choices=['profile', 'spacetime_density_m', 'spacetime_velocity_m', 'spacetime_density_c', 'spacetime_velocity_c', 'all'],
        help="List of plots to generate (default: all)."
    )
    parser.add_argument(
        '--output_dir',
        help="Directory to save the plots (default: the directory containing the input .npz file, or results/<scenario_name>/ if using --scenario_name)."
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help="Display plots interactively instead of just saving them."
    )
    parser.add_argument(
        '--no_save',
        action='store_true',
        help="Do not save the generated plots."
    )
    parser.add_argument(
        '--vmin',
        type=float,
        help="Minimum color scale value for spacetime plots."
    )
    parser.add_argument(
        '--vmax',
        type=float,
        help="Maximum color scale value for spacetime plots."
    )

    args = parser.parse_args()

    # Determine input file and default output directory
    default_output_dir = None
    if args.input:
        input_file = args.input
        if not os.path.exists(input_file):
            print(f"Error: Specified input file not found: {input_file}")
            sys.exit(1)
        print(f"Using specified input file: {input_file}")
        default_output_dir = os.path.dirname(input_file)
    else:
        # Input file not specified, scenario_name is required
        if not args.scenario_name:
            print("Error: Either --input or --scenario_name must be provided.")
            sys.exit(1)

        scenario_results_dir = os.path.join(args.results_dir, args.scenario_name)
        print(f"No input file specified. Searching for latest .npz in '{scenario_results_dir}'...")
        input_file = find_latest_npz(scenario_results_dir)
        if not input_file:
            print(f"Error: No .npz files found in '{scenario_results_dir}'.")
            sys.exit(1)
        print(f"Using latest file: {input_file}")
        default_output_dir = scenario_results_dir

    # Determine final output directory (use explicit override if provided)
    output_dir = args.output_dir if args.output_dir else default_output_dir
    if not output_dir: # Should not happen if logic above is correct, but safety check
        print("Error: Could not determine output directory.")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    # Determine save flag
    save_plots = not args.no_save

    # Load data
    try:
        data = load_simulation_data(input_file)
        times = data['times']
        states = data['states'] # Shape (num_times, 4, N_physical)
        grid = data['grid']
        params = data['params']
    except FileNotFoundError:
        print(f"Error: Could not load data. File not found: {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data from {input_file}: {e}")
        sys.exit(1)

    print(f"Data loaded for scenario: {params.scenario_name}")

    # Generate plots
    plots_to_generate = set(args.plots)
    if 'all' in plots_to_generate:
        plots_to_generate = {'profile', 'spacetime_density_m', 'spacetime_velocity_m', 'spacetime_density_c', 'spacetime_velocity_c'}

    print(f"Generating plots: {', '.join(sorted(list(plots_to_generate)))}")

    if 'profile' in plots_to_generate:
        print("  - Generating final time profile plot...")
        plot_profiles(
            state_physical=states[-1],
            grid=grid,
            time=times[-1],
            params=params,
            output_dir=output_dir,
            show=args.show,
            save=save_plots
        )

    if 'spacetime_density_m' in plots_to_generate:
        print("  - Generating spacetime density plot (motorcycles)...")
        plot_spacetime(
            times=times,
            states=states,
            grid=grid,
            params=params,
            variable='density',
            class_index=0, # Motorcycles
            output_dir=output_dir,
            show=args.show,
            save=save_plots,
            vmin=args.vmin,
            vmax=args.vmax
        )

    if 'spacetime_velocity_m' in plots_to_generate:
        print("  - Generating spacetime velocity plot (motorcycles)...")
        plot_spacetime(
            times=times,
            states=states,
            grid=grid,
            params=params,
            variable='velocity',
            class_index=0, # Motorcycles
            output_dir=output_dir,
            show=args.show,
            save=save_plots,
            vmin=args.vmin,
            vmax=args.vmax
        )

    if 'spacetime_density_c' in plots_to_generate:
        print("  - Generating spacetime density plot (cars)...")
        plot_spacetime(
            times=times,
            states=states,
            grid=grid,
            params=params,
            variable='density',
            class_index=2, # Cars
            output_dir=output_dir,
            show=args.show,
            save=save_plots,
            vmin=args.vmin,
            vmax=args.vmax
        )

    if 'spacetime_velocity_c' in plots_to_generate:
        print("  - Generating spacetime velocity plot (cars)...")
        plot_spacetime(
            times=times,
            states=states,
            grid=grid,
            params=params,
            variable='velocity',
            class_index=2, # Cars
            output_dir=output_dir,
            show=args.show,
            save=save_plots,
            vmin=args.vmin,
            vmax=args.vmax
        )

    print("Plot generation finished.")

if __name__ == "__main__":
    main()