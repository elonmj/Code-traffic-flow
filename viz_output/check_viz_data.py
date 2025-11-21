import pandas as pd
import numpy as np

INPUT_FILE = 'arz_model/data/fichier_de_travail_corridor_enriched.xlsx'

def check_data():
    df = pd.read_excel(INPUT_FILE)
    print("Columns:", df.columns)
    print("\nSample data:")
    print(df[['u_lon', 'u_lat', 'v_lon', 'v_lat', 'lanes_manual']].head())
    
    print("\nLanes stats:")
    print(df['lanes_manual'].describe())
    print("NaN lanes:", df['lanes_manual'].isna().sum())
    
    print("\nCoordinates bounds:")
    print("Lon:", df['u_lon'].min(), df['u_lon'].max())
    print("Lat:", df['u_lat'].min(), df['u_lat'].max())

if __name__ == "__main__":
    check_data()