# Traffic Dinguerie Visualization

This folder contains the "Dinguerie" - a high-performance 3D animated traffic visualization using PyDeck and Mapbox.

## Files

- `generate_kepler_map.py`: The Python script that generates the visualization.
- `traffic_dinguerie.html`: The generated HTML file containing the interactive map.

## How to Run

1. Ensure you have the required dependencies:
   ```bash
   pip install pydeck pandas numpy
   ```

2. Ensure your `.env` file has a valid `MAPBOX_API_KEY`.

3. Run the generation script:
   ```bash
   python viz_output/generate_kepler_map.py
   ```

4. Open `viz_output/traffic_dinguerie.html` in your web browser.

## Features

- **3D Road Network**: Visualized as a `PathLayer` with width based on lane count.
- **Animated Traffic**: Visualized as a `TripsLayer` with trails that move over time.
- **Synthetic Data**: Traffic is generated based on the road network topology (Lagos, Victoria Island).
- **Interactive**: Zoom, pan, and rotate (right-click drag) the map.

## Troubleshooting

- If the map is black, check your internet connection (Mapbox tiles need internet).
- If the animation doesn't start, check the browser console (F12) for errors.
- If "Waiting for deckInstance..." appears in the console, the map failed to initialize. Check your API key.
