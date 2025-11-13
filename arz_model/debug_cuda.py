import numba.cuda as cuda
print("Numba CUDA imported successfully")

print(f"cuda.is_available(): {cuda.is_available()}")

if cuda.is_available():
    print("CUDA is available!")
    try:
        device = cuda.get_current_device()
        print(f"Device: {device}")
        print(f"Name: {device.name}")
        print(f"Compute capability: {device.compute_capability}")
    except Exception as e:
        print(f"Error accessing device: {e}")
else:
    print("CUDA is NOT available.")

print("\n--- End of debug ---")