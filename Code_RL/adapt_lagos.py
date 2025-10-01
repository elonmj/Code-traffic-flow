"""
Adaptation des paramètres de trafic pour le contexte Victoria Island Lagos
"""

import yaml
import sys
import os

def create_realistic_traffic_params():
    """Créer des paramètres de trafic réalistes pour Victoria Island"""
    
    print("🚗 PARAMÈTRES DE TRAFIC VICTORIA ISLAND LAGOS")
    print("=" * 55)
    
    # Paramètres contextuels pour Lagos
    traffic_params = {
        'traffic': {
            'context': 'Victoria Island Lagos',
            'description': 'Paramètres adaptés au trafic urbain dense de Lagos',
            
            # Composition du trafic (spécifique à Lagos)
            'vehicle_mix': {
                'motorcycles_percentage': 35,  # Forte présence de motos à Lagos
                'cars_percentage': 45,
                'buses_percentage': 15,
                'trucks_percentage': 5
            },
            
            # Densités maximales (véh/km) - trafic dense Lagos
            'max_densities': {
                'motorcycles': 250,  # Motos très denses
                'cars': 120,
                'total': 370
            },
            
            # Vitesses libres (km/h) - ajustées pour Lagos
            'free_speeds': {
                'motorcycles': 32,  # Motos agiles dans le trafic
                'cars': 28,         # Voitures plus lentes en ville
                'average': 30
            },
            
            # Comportements spécifiques Lagos/Afrique de l'Ouest
            'behaviors': {
                'gap_filling_rate': 0.8,      # Comblement des espaces élevé
                'interweaving_rate': 0.7,     # Entrelacement fréquent
                'creeping_rate': 0.6,         # Avancement lent en file
                'patience_factor': 0.4,       # Patience limitée
                'signal_compliance': 0.7      # Respect modéré des signaux
            },
            
            # Paramètres temporels (secondes)
            'timing': {
                'reaction_time': 1.2,         # Temps de réaction moyen
                'startup_lost_time': 3.5,     # Temps perdu au démarrage
                'clearance_lost_time': 2.0,   # Temps perdu en fin de phase
                'yellow_time': 3.0,
                'all_red_time': 2.0
            }
        }
    }
    
    # Sauvegarder les paramètres de trafic
    with open('configs/traffic_lagos.yaml', 'w') as f:
        yaml.dump(traffic_params, f, default_flow_style=False, indent=2)
    
    print("✅ Paramètres de trafic Lagos créés")
    print("   📁 Fichier: configs/traffic_lagos.yaml")
    
    return traffic_params

def adapt_env_config_for_lagos():
    """Adapter la configuration d'environnement pour Lagos"""
    
    print("\n🌍 ADAPTATION ENVIRONNEMENT POUR LAGOS")
    print("=" * 45)
    
    # Configuration environnement adaptée
    env_config = {
        'environment': {
            'context': 'Victoria Island Lagos',
            'dt_decision': 10.0,           # Décisions RL toutes les 10s
            'episode_length': 3600,        # Episodes de 1h
            'max_steps': 360,              # 360 décisions par épisode
            
            # Normalisation adaptée au contexte Lagos
            'normalization': {
                'rho_max_motorcycles': 250,  # Densité max motos
                'rho_max_cars': 120,         # Densité max voitures
                'v_free_motorcycles': 32,    # Vitesse libre motos
                'v_free_cars': 28,           # Vitesse libre voitures
                'queue_max': 50,             # File max attendue
                'phase_time_max': 120        # Durée phase max
            },
            
            # Fonction de récompense adaptée
            'reward': {
                'w_wait_time': 1.2,          # Poids temps d'attente (important à Lagos)
                'w_queue_length': 0.6,       # Poids longueur files
                'w_stops': 0.4,              # Poids nombre d'arrêts
                'w_switch_penalty': 0.08,    # Pénalité changement (réduite)
                'w_throughput': 1.0,         # Poids débit (crucial)
                'reward_clip': [-5.0, 5.0],
                'stop_speed_threshold': 5.0  # Seuil vitesse arrêt (km/h)
            },
            
            # Observations adaptées
            'observation': {
                'ewma_alpha': 0.3,           # Lissage exponentiel
                'include_phase_timing': True,
                'include_queues': True,
                'include_speeds': True,
                'include_densities': True
            }
        }
    }
    
    # Sauvegarder
    with open('configs/env_lagos.yaml', 'w') as f:
        yaml.dump(env_config, f, default_flow_style=False, indent=2)
    
    print("✅ Configuration environnement Lagos créée")
    print("   📁 Fichier: configs/env_lagos.yaml")
    
    return env_config

def adapt_signals_for_lagos():
    """Adapter la configuration des signaux pour Lagos"""
    
    print("\n🚦 ADAPTATION SIGNAUX POUR LAGOS")
    print("=" * 40)
    
    signals_config = {
        'signals': {
            'context': 'Victoria Island Lagos',
            'intersection_type': 'urban_arterial',
            
            # Phases adaptées aux intersections Lagos
            'phases': [
                {
                    'id': 0,
                    'name': 'akin_adesola_ns',
                    'description': 'Akin Adesola Nord-Sud vert',
                    'movements': ['north_through', 'south_through', 'north_left', 'south_left']
                },
                {
                    'id': 1,
                    'name': 'adeola_odeku_ew',  
                    'description': 'Adeola Odeku Est-Ouest vert',
                    'movements': ['east_through', 'west_through', 'east_left', 'west_left']
                }
            ],
            
            # Contraintes temporelles adaptées Lagos
            'timing_constraints': {
                'min_green': 15.0,           # Vert minimum plus long (trafic dense)
                'max_green': 90.0,           # Vert maximum réduit (fluidité)
                'yellow_time': 4.0,          # Jaune plus long (sécurité)
                'all_red_time': 3.0,         # Rouge général plus long
                'cycle_time_min': 60.0,      # Cycle minimum
                'cycle_time_max': 120.0      # Cycle maximum
            },
            
            # Paramètres de sécurité
            'safety': {
                'pedestrian_clearance': 8.0, # Dégagement piétons
                'vehicle_clearance': 4.0,    # Dégagement véhicules
                'startup_lost_time': 3.5,    # Temps perdu démarrage
                'clearance_lost_time': 2.5   # Temps perdu dégagement
            }
        }
    }
    
    # Sauvegarder
    with open('configs/signals_lagos.yaml', 'w') as f:
        yaml.dump(signals_config, f, default_flow_style=False, indent=2)
    
    print("✅ Configuration signaux Lagos créée")  
    print("   📁 Fichier: configs/signals_lagos.yaml")
    
    return signals_config

def create_lagos_config_set():
    """Créer un ensemble complet de configurations pour Lagos"""
    
    print("\n📦 CRÉATION ENSEMBLE CONFIGURATIONS LAGOS")
    print("=" * 50)
    
    # Créer toutes les configurations
    traffic_params = create_realistic_traffic_params()
    env_config = adapt_env_config_for_lagos()
    signals_config = adapt_signals_for_lagos()
    
    # Configuration maître pour utiliser les configs Lagos
    master_config = {
        'experiment': {
            'name': 'Lagos_Victoria_Island',
            'description': 'Contrôle de signaux RL pour Victoria Island Lagos',
            'location': 'Lagos, Nigeria',
            'configs': {
                'network': 'network_real.yaml',
                'traffic': 'traffic_lagos.yaml', 
                'environment': 'env_lagos.yaml',
                'signals': 'signals_lagos.yaml',
                'endpoint': 'endpoint.yaml'  # Garder endpoint standard
            },
            'parameters': {
                'algorithm': 'DQN',
                'timesteps': 50000,
                'evaluation_freq': 5000,
                'save_freq': 10000
            }
        }
    }
    
    with open('configs/lagos_master.yaml', 'w') as f:
        yaml.dump(master_config, f, default_flow_style=False, indent=2)
    
    print("\n🎯 ENSEMBLE COMPLET CRÉÉ !")
    print("   📁 Fichiers créés :")
    print("      • configs/traffic_lagos.yaml")
    print("      • configs/env_lagos.yaml")
    print("      • configs/signals_lagos.yaml")
    print("      • configs/lagos_master.yaml")
    
    print("\n🚀 Utilisation :")
    print("   • Test: python demo.py (avec configs Lagos)")
    print("   • Train: python train.py --config lagos --use-mock")
    
    return master_config

def main():
    """Création principale des configurations Lagos"""
    create_lagos_config_set()

if __name__ == "__main__":
    main()
