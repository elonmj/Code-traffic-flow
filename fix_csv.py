#!/usr/bin/env python3
"""Fix the corrupted CSV file by removing extra fields."""

import pandas as pd
import sys

csv_file = 'donnees_trafic_75_segments (2).csv'

try:
    # Read CSV with on_bad_lines='skip' to handle corrupted rows
    df = pd.read_csv(csv_file, on_bad_lines='skip')
    print(f"✅ CSV loaded successfully (bad lines skipped)")
    print(f"   Rows: {len(df)}")
    print(f"   Columns ({len(df.columns)}): {list(df.columns)}")
    
    # Save corrected CSV
    df.to_csv(csv_file, index=False)
    print(f"✅ CSV saved successfully")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
