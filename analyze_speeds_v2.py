import pandas as pd
import numpy as np

df = pd.read_csv('Code_RL/data/donnees_vitesse_historique.csv')

print("📊 DONNEES VITESSE HISTORIQUES - VICTORIA ISLAND")
print("="*70)
print(f"\n🔹 Current Speed (observed, km/h):")
print(df['current_speed'].describe())

print(f"\n🔹 Freeflow Speed (km/h):")
print(df['freeflow_speed'].describe())

print(f"\n💡 ANALYSE:")
print(f"  - Vitesse moyenne observée : {df['current_speed'].mean():.2f} km/h")
print(f"  - Vitesse free-flow moyenne: {df['freeflow_speed'].mean():.2f} km/h")
print(f"  - Ratio (current/freeflow):  {df['current_speed'].mean() / df['freeflow_speed'].mean():.1%}")
print(f"  - Congestion level:          {100 * (1 - df['current_speed'].mean() / df['freeflow_speed'].mean()):.1f}%")

print(f"\n⚙️  PARAMETRES MODELE ACTUELS:")
print(f"  V_c (motos)    = 30.0 km/h")
print(f"  V_m (voitures) = 60.0 km/h")
print(f"  v_max (limite) = 50.0 km/h (déduit de speed_limit)")

print(f"\n🎯 PARAMETRES RECOMMANDES:")
avg_current = df['current_speed'].mean()
avg_freeflow = df['freeflow_speed'].mean()

# Adjusted equilibrium speeds to match observed
print(f"  V_c (motos)    = {avg_current * 0.8:.1f} km/h  (80% de vitesse observée)")
print(f"  V_m (voitures) = {avg_current * 1.2:.1f} km/h  (120% de vitesse observée)")
print(f"  v_max (limite) = {avg_freeflow:.1f} km/h  (vitesse freeflow observée)")
print(f"\n📌 Densités initiales doivent correspondre à congestion modérée !")
