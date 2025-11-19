from arz_model.config.network_simulation_config import NodeConfig
from arz_model.network.node import Node

print("Testing Node.from_config with traffic lights...")

try:
    node_config = NodeConfig(
        id="test_node",
        type="signalized",
        position=[0.0, 0.0],
        incoming_segments=["seg1"],
        outgoing_segments=["seg2"],
        traffic_light_config={
            "cycle_time": 60.0,
            "green_time": 30.0
        }
    )

    node = Node.from_config(node_config)

    if node.traffic_lights:
        print("✅ Traffic lights initialized!")
        print(f"Cycle time: {node.traffic_lights.cycle_time}")
        
        # Test get_current_green_segments
        green = node.traffic_lights.get_current_green_segments(10.0)
        print(f"t=10.0 (Green): {green}")
        
        red = node.traffic_lights.get_current_green_segments(40.0)
        print(f"t=40.0 (Red): {red}")
        
    else:
        print("❌ Traffic lights NOT initialized!")

except Exception as e:
    print(f"❌ Error: {e}")
