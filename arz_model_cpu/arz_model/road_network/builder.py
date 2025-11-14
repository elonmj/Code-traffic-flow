"""
Builds the simulation-specific network objects using the NetworkGrid API.

This module acts as a factory, correctly using the `add_segment` and `add_node`
methods of the NetworkGrid to construct the simulation environment from the
validated Pydantic models.
"""
from collections import defaultdict
from typing import Tuple
import sys
import os

# Add parent directory to path to enable absolute imports
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from arz_model.road_network.models import RoadNetwork
from arz_model.network.network_grid import NetworkGrid
from arz_model.core.parameters import ModelParameters
from arz_model.core.intersection import Intersection
from arz_model.core.traffic_lights import TrafficLightController, Phase
from arz_model.config.network_simulation_config import NetworkSimulationConfig, NodeConfig, SegmentConfig, LinkConfig
from arz_model.config.time_config import TimeConfig
from arz_model.config.physics_config import PhysicsConfig
from arz_model.config.ic_config import ICConfig
from arz_model.config.bc_config import BoundaryConditionsConfig
from arz_model.config.grid_config import GridConfig

def build_simulation_network(
    road_network: RoadNetwork,
    time_config: TimeConfig,
    physics_config: PhysicsConfig,
    ic_config: ICConfig,
    bc_config: BoundaryConditionsConfig,
    grid_config: GridConfig,
    default_dx: float = 5.0
) -> Tuple[NetworkGrid, NetworkSimulationConfig]:
    """
    Builds a NetworkGrid object and its corresponding NetworkSimulationConfig
    from a RoadNetwork model using a multi-pass approach.
    
    Returns a tuple: (NetworkGrid, NetworkSimulationConfig)
    """
    node_ids = set(road_network.nodes.keys())
    node_configs = {node_id: {'incoming': [], 'outgoing': []} for node_id in node_ids}

    # Pass 1: Determine connectivity for all nodes from the links data.
    print("--- Pass 1: Determining Node Connectivity ---")
    for i, link_model in enumerate(road_network.links):
        segment_id = f"seg_{i}"
        # Ensure nodes from links are in the config, even if not in the nodes list (defensive)
        if link_model.u not in node_configs:
            node_configs[link_model.u] = {'incoming': [], 'outgoing': []}
        if link_model.v not in node_configs:
            node_configs[link_model.v] = {'incoming': [], 'outgoing': []}
            
        node_configs[link_model.u]['outgoing'].append(segment_id)
        node_configs[link_model.v]['incoming'].append(segment_id)
    print(f"Node connectivity determined for {len(node_configs)} nodes.")

    boundary_conditions_config = {'boundary_conditions': {}}
    junctions = []
    
    # Pass 2: Classify nodes and build configurations for boundaries and junctions.
    print("\n--- Pass 2: Identifying Node Types and Building Configs ---")
    source_nodes, sink_nodes, junction_nodes, isolated_nodes = [], [], [], []
    
    pydantic_nodes: Dict[str, NodeConfig] = {}
    pydantic_segments: Dict[str, SegmentConfig] = {}

    for node_id, config in node_configs.items():
        is_source = not config['incoming'] and config['outgoing']
        is_sink = config['incoming'] and not config['outgoing']
        is_junction = config['incoming'] and config['outgoing']

        if is_source:
            source_nodes.append(node_id)
            print(f"Node {node_id}: Identified as SOURCE. Outgoing to: {config['outgoing']}")
            for segment_id in config['outgoing']:
                if segment_id not in boundary_conditions_config['boundary_conditions']:
                    boundary_conditions_config['boundary_conditions'][segment_id] = {}
                # Define inflow boundary condition at the start of the segment
                boundary_conditions_config['boundary_conditions'][segment_id]['start'] = {
                    'type': 'inflow',
                    'value': 0.1  # Example: constant inflow, can be parameterized later
                }
        elif is_sink:
            sink_nodes.append(node_id)
            print(f"Node {node_id}: Identified as SINK. Incoming from: {config['incoming']}")
            for segment_id in config['incoming']:
                if segment_id not in boundary_conditions_config['boundary_conditions']:
                    boundary_conditions_config['boundary_conditions'][segment_id] = {}
                # Define free outflow boundary condition at the end of the segment
                boundary_conditions_config['boundary_conditions'][segment_id]['end'] = {
                    'type': 'outflow',
                    'value': None
                }
        elif is_junction:
            junction_nodes.append(node_id)
            print(f"Node {node_id}: Identified as JUNCTION. Incoming: {config['incoming']}, Outgoing: {config['outgoing']}")
            # Prepare Intersection object for later
            junctions.append(
                Intersection(
                    node_id=str(node_id),
                    incoming_segments=config['incoming'],
                    outgoing_segments=config['outgoing'],
                    traffic_lights=TrafficLightController(
                        cycle_time=60,
                        phases=[]
                    )
                )
            )
            pydantic_nodes[str(node_id)] = NodeConfig(
                type='signalized', # Assume signalized for now
                incoming_segments=config['incoming'],
                outgoing_segments=config['outgoing']
            )
        else:
            isolated_nodes.append(node_id)
            print(f"Node {node_id}: Identified as ISOLATED. No connections.")

    print(f"\nSummary: {len(source_nodes)} sources, {len(sink_nodes)} sinks, {len(junction_nodes)} junctions, {len(isolated_nodes)} isolated.")
    
    # Pass 2.5: Build Pydantic Node and Segment Configs
    for node_id, config in node_configs.items():
        is_junction = config['incoming'] and config['outgoing']
        node_type = 'signalized' if is_junction else 'boundary'
        pydantic_nodes[str(node_id)] = NodeConfig(
            type=node_type,
            position=road_network.nodes[node_id].position if node_id in road_network.nodes else [0,0],
            incoming_segments=config['incoming'],
            outgoing_segments=config['outgoing']
        )

    for i, link_model in enumerate(road_network.links):
        segment_id = f"seg_{i}"
        length = link_model.length_m
        num_cells = max(10, int(length / default_dx))
        
        pydantic_segments[segment_id] = SegmentConfig(
            x_min=0.0,
            x_max=length,
            N=num_cells,
            start_node=str(link_model.u),
            end_node=str(link_model.v)
        )

    # Pass 2.7: Build Pydantic Link Configs from junctions
    pydantic_links: Dict[str, LinkConfig] = {}
    link_counter = 0
    for node_id in junction_nodes:
        config = node_configs[node_id]
        incoming = config['incoming']
        outgoing = config['outgoing']
        # This logic is flawed. It creates a link from *every* incoming
        # to *every* outgoing segment, causing duplicates.
        # A real implementation would use turning movement data.
        # For now, let's just link the first incoming to the first outgoing.
        if incoming and outgoing:
            in_seg = incoming[0]
            out_seg = outgoing[0]
            link_id = f"link_{link_counter}"
            pydantic_links[link_id] = LinkConfig(
                from_segment=in_seg,
                to_segment=out_seg,
                via_node=str(node_id)
            )
            link_counter += 1

    # Create the final simulation config object
    final_config = NetworkSimulationConfig(
        time=time_config,
        physics=physics_config,
        grid=grid_config,
        ic=ic_config,
        segments=pydantic_segments,
        nodes=pydantic_nodes,
        links=pydantic_links,
        boundary_conditions=bc_config
    )

    # Pass 3: Construct the NetworkGrid and populate it.
    print("\n--- Pass 3: Constructing and Populating the NetworkGrid ---")
    simulation_network = NetworkGrid(network_id="urban_network", simulation_config=final_config, grid_config=grid_config)

    # Add all nodes first, without position data initially
    for node_id in node_configs.keys():
        simulation_network.add_node(node_id=str(node_id))
    print(f"Added {len(simulation_network.nodes)} nodes to the network.")

    # Add all segments
    for i, link_model in enumerate(road_network.links):
        segment_id = f"seg_{i}"
        num_lanes = link_model.lanes
        
        # Use a default dx, can be refined later
        N = max(10, int(link_model.length_m / default_dx))

        simulation_network.add_segment(
            segment_id=segment_id,
            xmin=0,
            xmax=link_model.length_m,
            N=N,
            start_node=str(link_model.u),
            end_node=str(link_model.v)
        )
    print(f"Added {len(simulation_network.segments)} segments to the network.")

    # Update nodes with their final connectivity information
    for node_id, node in simulation_network.nodes.items():
        config = node_configs.get(int(node_id))
        if config:
            node.incoming_segments = config['incoming']
            node.outgoing_segments = config['outgoing']
            
            is_source = not config['incoming'] and config['outgoing']
            is_sink = config['incoming'] and not config['outgoing']
            is_junction = config['incoming'] and config['outgoing']

            if is_junction:
                # Find the corresponding Intersection object and assign it
                junction_obj = next((j for j in junctions if j.node_id == node_id), None)
                if junction_obj:
                    node.intersection = junction_obj
                    node.traffic_lights = junction_obj.traffic_lights
            
            # Set node type for clarity, although this is not strictly used by the core logic
            if is_junction:
                node.node_type = 'junction'
            elif is_source:
                node.node_type = 'source'
            elif is_sink:
                node.node_type = 'sink'

    print(f"Configured {len(junctions)} junctions and updated all node connectivity.")

    # Final validation
    simulation_network.initialize()
    print("--- NetworkGrid Initialized Successfully ---")

    return simulation_network, final_config