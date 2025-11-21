import pandas as pd
import pydeck as pdk
import numpy as np
import math
import random
import os

# Configuration
INPUT_FILE = 'arz_model/data/fichier_de_travail_corridor_enriched.xlsx'
OUTPUT_HTML = 'viz_output/traffic_dinguerie.html'
ENV_FILE = '.env'

def load_mapbox_token():
    """Loads Mapbox token from .env file."""
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, 'r') as f:
            for line in f:
                if line.strip().startswith('MAPBOX_API_KEY='):
                    return line.strip().split('=', 1)[1]
    return None

def generate_synthetic_trips(df, duration_seconds=1800):
    """
    Generates synthetic vehicle trajectories (trips) for visualization.
    """
    trips = []
    
    print(f"Generating synthetic traffic for {len(df)} segments...")
    
    for _, row in df.iterrows():
        if pd.isna(row['u_lat']) or pd.isna(row['v_lat']):
            continue
            
        start = [row['u_lon'], row['u_lat']]
        end = [row['v_lon'], row['v_lat']]
        
        # Determine number of vehicles based on lanes (synthetic density)
        lanes = row.get('lanes_manual', 1)
        if pd.isna(lanes): lanes = 1
        lanes = int(lanes) # Ensure integer
        
        # Traffic intensity (randomized for "organic" look)
        intensity = int(lanes * 5) 
        
        # Speed (km/h to m/s approx)
        speed_kmh = row.get('maxspeed_manual_kmh', 30)
        if pd.isna(speed_kmh): speed_kmh = 30
        
        # Calculate segment length (approx)
        dist = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2) * 111000 # degrees to meters
        travel_time = dist / (speed_kmh / 3.6)
        
        if travel_time == 0: continue

        # Generate vehicles
        for i in range(intensity):
            # Random start time within the duration
            start_time = random.uniform(0, duration_seconds)
            
            # Create a trajectory: [x, y] and separate timestamps
            # Simple linear interpolation
            path_coords = []
            timestamps = []
            steps = 10
            for s in range(steps + 1):
                t = s / steps
                lon = start[0] + (end[0] - start[0]) * t
                lat = start[1] + (end[1] - start[1]) * t
                timestamp = start_time + (travel_time * t)
                path_coords.append([lon, lat])
                timestamps.append(timestamp)
            
            # Add the main trip
            trips.append({"path": path_coords, "timestamps": timestamps, "vendor": 0})
            
            # Add "Ghost Trips" for seamless looping
            # 1. Previous loop (ends in current window)
            timestamps_prev = [t - duration_seconds for t in timestamps]
            trips.append({"path": path_coords, "timestamps": timestamps_prev, "vendor": 0})
            
            # 2. Next loop (starts in current window)
            timestamps_next = [t + duration_seconds for t in timestamps]
            trips.append({"path": path_coords, "timestamps": timestamps_next, "vendor": 0})

    return trips

def main():
    # 1. Load Data
    print("Loading network data...")
    df = pd.read_excel(INPUT_FILE)
    
    # 2. Generate Trips
    trips_data = generate_synthetic_trips(df)
    print(f"Generated {len(trips_data)} vehicle trajectories.")
    
    # 3. Configure PyDeck
    mapbox_key = load_mapbox_token()
    if mapbox_key:
        print("Mapbox token loaded.")
        pdk.settings.mapbox_key = mapbox_key
    else:
        print("WARNING: Mapbox token not found in .env. Map might not render correctly.")

    view_state = pdk.ViewState(
        latitude=6.43,
        longitude=3.42,
        zoom=13,
        pitch=45,
        bearing=0
    )

    # Layer 1: Road Network (PathLayer) - The "Base"
    # We need a clean dataframe for paths
    paths = []
    for _, row in df.iterrows():
        if pd.isna(row['u_lat']) or pd.isna(row['v_lat']): continue
        
        lanes = row.get('lanes_manual', 1)
        if pd.isna(lanes): lanes = 1
        lanes = int(lanes)
        
        paths.append({
            "path": [[row['u_lon'], row['u_lat']], [row['v_lon'], row['v_lat']]],
            "lanes": lanes,
            "width": lanes * 2
        })
    
    layer_roads = pdk.Layer(
        "PathLayer",
        paths,
        get_path="path",
        get_width="width",
        get_color=[50, 50, 50],
        width_min_pixels=2
    )

    # Layer 2: Traffic Flow (TripsLayer) - The "Dinguerie"
    layer_trips = pdk.Layer(
        "TripsLayer",
        trips_data,
        id="traffic-trips",
        get_path="path",
        get_timestamps="timestamps",
        get_color=[253, 128, 93],
        opacity=0.8,
        width_min_pixels=3,
        trail_length=50,
        current_time=0
    )

    # 4. Render
    r = pdk.Deck(
        map_provider="mapbox",
        layers=[layer_roads, layer_trips],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/navigation-night-v1",
        tooltip=True,
        api_keys={'mapbox': mapbox_key}
    )
    
    # Animation configuration
    # PyDeck HTML export with animation requires specific handling or it just loops
    # For TripsLayer, we usually need to update 'current_time' in a loop in Python (Jupyter)
    # OR use the standalone HTML animation support.
    # PyDeck's to_html creates a standalone file. 
    # To animate TripsLayer in standalone HTML, we need to inject JS or use a specific PyDeck feature.
    # Actually, PyDeck 0.8+ supports animation in to_html if we don't specify a loop?
    # Let's try basic export. If it doesn't animate, it's still a cool static view of trails.
    
    print(f"Exporting to {OUTPUT_HTML}...")
    r.to_html(OUTPUT_HTML)
    
    # Inject animation script
    inject_animation_script(OUTPUT_HTML)
    
    print("Done!")

def inject_animation_script(html_path):
    """Injects JavaScript to animate the TripsLayer."""
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Inject Mapbox CSS
    mapbox_css = '<link href="https://api.tiles.mapbox.com/mapbox-gl-js/v1.13.0/mapbox-gl.css" rel="stylesheet" />'
    content = content.replace('<head>', f'<head>\n    {mapbox_css}')

    # 1. Expose deckInstance to window and add error handling
    content = content.replace('const deckInstance = createDeck', 'try { window.deckInstance = createDeck')
    
    # Close the try block
    # We look for the end of the createDeck call which is followed by the closing script tag
    # The pattern in the file is:
    #     });
    # 
    #   </script>
    content = content.replace('    });\n\n  </script>', '    });\n    } catch(error) { console.error("Deck creation failed:", error); }\n  </script>')
    
    # 2. Add animation loop
    animation_script = """
    <script>
      console.log("Animation script loaded");
      // Animation Loop for TripsLayer
      const loopLength = 1800; // 30 minutes
      const animationSpeed = 30; // 30x speed

      function animate() {
        const currentTime = (Date.now() / 1000 * animationSpeed) % loopLength;
        
        if (window.deckInstance) {
            // We need to clone the layer with new currentTime
            // Note: This assumes the layer structure matches what we expect
            const layers = window.deckInstance.props.layers.map(layer => {
                if (layer.id === 'traffic-trips') {
                    return layer.clone({currentTime: currentTime});
                }
                return layer;
            });
            
            window.deckInstance.setProps({layers: layers});
        } else {
            console.log("Waiting for deckInstance...");
        }
        requestAnimationFrame(animate);
      }
      
      // Start animation after a short delay to ensure map is loaded
      setTimeout(animate, 1000);
    </script>
    """
    
    # Remove original body close to append script before html close
    content = content.replace('</body>', '') 
    content = content.replace('</html>', f'{animation_script}\n</body>\n</html>')
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Animation script injected.")

if __name__ == "__main__":
    main()
