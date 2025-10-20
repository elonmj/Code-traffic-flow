"""
Convert all test figures to PNG format for LaTeX integration.
Also regenerate with PNG output directly.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import all test modules
from test_riemann_motos_shock import run_test as test1
from test_riemann_motos_rarefaction import run_test as test2
from test_riemann_voitures_shock import run_test as test3
from test_riemann_voitures_rarefaction import run_test as test4
from test_riemann_multiclass import run_test as test5
from convergence_study import run_convergence_study

print("=" * 80)
print("REGENERATING ALL FIGURES IN PNG FORMAT")
print("=" * 80)

# Modify matplotlib to save PNG by default
import matplotlib.pyplot as plt
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['savefig.dpi'] = 300

tests = [
    ("Test 1 - Shock motos", test1),
    ("Test 2 - Rarefaction motos", test2),
    ("Test 3 - Shock voitures", test3),
    ("Test 4 - Rarefaction voitures", test4),
    ("Test 5 - Multiclass CRITIQUE", test5),
]

print("\n🖼️  Generating PNG figures...")
for name, test_func in tests:
    print(f"\n  → {name}")
    try:
        test_func(save_results=True)
        print(f"    ✅ Done")
    except Exception as e:
        print(f"    ❌ Error: {e}")

print(f"\n  → Convergence study")
try:
    run_convergence_study(save_results=True)
    print(f"    ✅ Done")
except Exception as e:
    print(f"    ❌ Error: {e}")

print("\n" + "=" * 80)
print("✅ ALL PNG FIGURES GENERATED")
print("=" * 80)

# List generated files
output_dir = project_root / "figures" / "niveau1_riemann"
png_files = list(output_dir.glob("*.png"))
print(f"\n📂 Generated {len(png_files)} PNG files:")
for f in sorted(png_files):
    print(f"  - {f.name}")

print("\n💡 Ces fichiers PNG peuvent être inclus directement dans LaTeX avec:")
print("   \\includegraphics[width=0.85\\textwidth]{figures/niveau1_riemann/test1_shock_motos.png}")
