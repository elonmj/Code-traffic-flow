#!/usr/bin/env python
"""Inspect kernel metadata structure"""
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Get one kernel and inspect it
kernels = api.kernels_list(page_size=5, sort_by='dateRun')
if kernels:
    k = kernels[0]
    print("📋 Kernel metadata structure:")
    print(f"Ref: {k.ref}\n")
    
    # List all attributes
    for attr in dir(k):
        if not attr.startswith('_'):
            try:
                val = getattr(k, attr)
                if not callable(val):
                    print(f"  {attr}: {val}")
            except:
                pass
