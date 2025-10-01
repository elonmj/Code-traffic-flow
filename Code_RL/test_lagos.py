"""
Test de la configuration Lagos avec données réelles
"""

import sys
import os

# Add src to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

def test_lagos_config():
    """Test de la configuration Lagos"""
    print("🌍 TEST CONFIGURATION LAGOS VICTORIA ISLAND")
    print("=" * 55)
    
    try:
        from utils.config import load_config
        from endpoint.client import create_endpoint_client, EndpointConfig
        from signals.controller import create_signal_controller
        from env.traffic_signal_env import TrafficSignalEnv, EnvironmentConfig
        
        print("   ✓ Imports réussis")
        
        # Charger les configs Lagos
        print("\n📁 Chargement configurations Lagos...")
        
        configs = {}
        try:
            configs["endpoint"] = load_config("configs/endpoint.yaml")
            configs["network"] = load_config("configs/network_real.yaml") 
            configs["env"] = load_config("configs/env_lagos.yaml")
            configs["signals"] = load_config("configs/signals_lagos.yaml")
            print("   ✓ Toutes les configs Lagos chargées")
        except FileNotFoundError as e:
            print(f"   ⚠️  Fichier manquant: {e}")
            return False
        
        # Afficher les spécificités Lagos
        print(f"\n🚗 Paramètres trafic Lagos:")
        env_config = configs["env"]["environment"]
        print(f"   • Densité max motos: {env_config['normalization']['rho_max_motorcycles']}")
        print(f"   • Densité max voitures: {env_config['normalization']['rho_max_cars']}")
        print(f"   • Vitesse libre motos: {env_config['normalization']['v_free_motorcycles']} km/h")
        print(f"   • Vitesse libre voitures: {env_config['normalization']['v_free_cars']} km/h")
        
        # Afficher les récompenses adaptées
        print(f"\n💰 Fonction récompense Lagos:")
        reward_config = env_config["reward"]
        print(f"   • Poids temps attente: {reward_config['w_wait_time']} (élevé)")
        print(f"   • Poids files: {reward_config['w_queue_length']}")
        print(f"   • Poids débit: {reward_config['w_throughput']} (important)")
        
        # Signaux adaptés
        print(f"\n🚦 Configuration signaux Lagos:")
        signals_config = configs["signals"]["signals"]
        timing = signals_config["timings"]
        print(f"   • Vert minimum: {timing['min_green']}s (adapté trafic dense)")
        print(f"   • Vert maximum: {timing['max_green']}s")
        print(f"   • Jaune: {timing['yellow']}s (sécurité)")
        print(f"   • Rouge général: {timing['all_red']}s")
        
        # Test création environnement Lagos
        print(f"\n🔄 Création environnement Lagos...")
        
        # Endpoint avec filtrage des clés
        endpoint_params = configs["endpoint"]["endpoint"]
        allowed_keys = {"protocol", "host", "port", "base_url", "dt_sim", "timeout", "max_retries", "retry_backoff"}
        endpoint_params_filtered = {k: v for k, v in endpoint_params.items() if k in allowed_keys}
        
        endpoint_config = EndpointConfig(**endpoint_params_filtered)
        endpoint_config.protocol = "mock"  # Force mock
        endpoint = create_endpoint_client(endpoint_config)
        print("   ✓ Client endpoint Lagos créé")
        
        # Signal controller
        controller = create_signal_controller(configs["signals"])
        print("   ✓ Contrôleur signaux Lagos créé")
        
        # Environment config
        env_config_data = configs["env"]["environment"]
        env_config_obj = EnvironmentConfig(
            dt_decision=env_config_data["dt_decision"],
            episode_length=env_config_data["episode_length"],
            max_steps=env_config_data["max_steps"],
            rho_max_motorcycles=env_config_data["normalization"]["rho_max_motorcycles"],
            rho_max_cars=env_config_data["normalization"]["rho_max_cars"],
            v_free_motorcycles=env_config_data["normalization"]["v_free_motorcycles"],
            v_free_cars=env_config_data["normalization"]["v_free_cars"],
            queue_max=env_config_data["normalization"]["queue_max"],
            phase_time_max=env_config_data["normalization"]["phase_time_max"],
            w_wait_time=env_config_data["reward"]["w_wait_time"],
            w_queue_length=env_config_data["reward"]["w_queue_length"],
            w_stops=env_config_data["reward"]["w_stops"],
            w_switch_penalty=env_config_data["reward"]["w_switch_penalty"],
            w_throughput=env_config_data["reward"]["w_throughput"],
            reward_clip=tuple(env_config_data["reward"]["reward_clip"]),
            stop_speed_threshold=env_config_data["reward"]["stop_speed_threshold"],
            ewma_alpha=env_config_data["observation"]["ewma_alpha"],
            include_phase_timing=env_config_data["observation"]["include_phase_timing"],
            include_queues=env_config_data["observation"]["include_queues"]
        )
        
        # Branch IDs from network
        try:
            branch_ids = [branch["id"] for branch in configs["network"]["network"]["branches"]]
            print(f"   ✓ {len(branch_ids)} branches réseau réel chargées")
        except KeyError:
            # Fallback si network_real.yaml n'existe pas encore
            branch_ids = ["north_in", "south_in", "east_in", "west_in"]
            print(f"   ⚠️  Utilisation branches par défaut (network_real.yaml manquant)")
        
        # Create environment
        env = TrafficSignalEnv(
            endpoint_client=endpoint,
            signal_controller=controller,
            config=env_config_obj,
            branch_ids=branch_ids
        )
        print("   ✓ Environnement Lagos créé avec succès")
        
        # Test rapide
        print(f"\n🎯 Test rapide environnement Lagos:")
        obs, info = env.reset(seed=42)
        print(f"   ✓ Reset: obs shape {obs.shape}")
        print(f"   ✓ Phase initiale: {info.get('phase_id', 'N/A')}")
        
        # Test 3 étapes
        total_reward = 0
        for step in range(3):
            action = step % 2  # Alterne les actions
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            print(f"   ✓ Step {step+1}: action={action}, reward={reward:.3f}")
            
            if done or truncated:
                break
        
        print(f"   ✓ Récompense totale: {total_reward:.3f}")
        
        env.close()
        print("   ✓ Environnement fermé")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    success = test_lagos_config()
    
    if success:
        print(f"\n🎉 CONFIGURATION LAGOS OPÉRATIONNELLE !")
        print(f"   ✅ Toutes les configs Lagos fonctionnent")
        print(f"   ✅ Paramètres adaptés au contexte Victoria Island")
        print(f"   ✅ Environnement RL prêt")
        
        print(f"\n🚀 Commandes pour utiliser Lagos:")
        print(f"   • Test: python demo.py (mettre config Lagos par défaut)")  
        print(f"   • Train: python train.py --config lagos --use-mock --timesteps 5000")
        print(f"   • Analyse: python analyze_corridor.py")
    else:
        print(f"\n❌ Problèmes avec la configuration Lagos")
        print(f"   Vérifiez que tous les fichiers de config existent")

if __name__ == "__main__":
    main()
