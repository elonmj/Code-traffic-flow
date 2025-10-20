"""Check Kaggle kernel status"""
from kaggle.api.kaggle_api_extended import KaggleApi
from datetime import datetime

api = KaggleApi()
api.authenticate()

# List recent kernels
kernels = api.kernels_list(user='joselonm', sort_by='dateRun')
print("=" * 70)
print("Recent Kaggle Kernels:")
print("=" * 70)
for i, k in enumerate(kernels[:5]):
    print(f"{i}: {k.ref} (v{k.current_version_number})")
    if hasattr(k, 'date_created'):
        print(f"   Date: {k.date_created}")

print("\n" + "=" * 70)
print("Checking most recent kernel:")
print("=" * 70)

if kernels:
    recent = kernels[0]
    print(f"Ref: {recent.ref}")
    print(f"Version: {recent.current_version_number}")
    
    try:
        status_info = api.kernels_status(recent.ref)
        print(f"\nStatus info type: {type(status_info)}")
        print(f"Status info: {status_info}")
        
        # Try to extract status
        if hasattr(status_info, 'status'):
            print(f"\n✅ Has .status attribute: {status_info.status}")
        if hasattr(status_info, 'failureMessage'):
            print(f"   Failure message: {status_info.failureMessage}")
    except Exception as e:
        print(f"❌ Status error: {type(e).__name__}: {e}")
