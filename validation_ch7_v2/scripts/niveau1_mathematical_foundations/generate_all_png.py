"""
Quick PNG generation script - regenerates all figures as PNG for LaTeX.
Uses the existing test scripts with modified matplotlib backend.
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set PNG as default format
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['savefig.dpi'] = 300

print("=" * 80)
print("GENERATING ALL PNG FIGURES FOR LATEX")
print("=" * 80)

# Import and patch the save functions
import test_riemann_motos_rarefaction as test2
import test_riemann_voitures_shock as test3
import test_riemann_voitures_rarefaction as test4
import test_riemann_multiclass as test5
import convergence_study

# Patch to save PNG
original_savefig = plt.savefig

def savefig_png(path, **kwargs):
    """Save both PNG and PDF."""
    path = Path(path)
    png_path = path.with_suffix('.png')
    pdf_path = path.with_suffix('.pdf')
    original_savefig(png_path, **kwargs)
    original_savefig(pdf_path, **kwargs)
    return png_path, pdf_path

plt.savefig = savefig_png

# Run tests
tests = [
    ("Test 2 - Rarefaction motos", test2.run_test),
    ("Test 3 - Shock voitures", test3.run_test),
    ("Test 4 - Rarefaction voitures", test4.run_test),
    ("Test 5 - Multiclass", test5.run_test),
    ("Convergence study", convergence_study.run_convergence_study),
]

for name, func in tests:
    print(f"\n🖼️  {name}")
    try:
        func(save_results=True)
        print(f"   ✅ PNG + PDF generated")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 80)
print("✅ ALL FIGURES GENERATED")
print("=" * 80)

# List PNG files
output_dir = project_root / "figures" / "niveau1_riemann"
png_files = sorted(output_dir.glob("*.png"))
print(f"\n📂 {len(png_files)} PNG files ready for LaTeX:")
for f in png_files:
    size_kb = f.stat().st_size / 1024
    print(f"   • {f.name} ({size_kb:.1f} KB)")

# Copy to deliverables
deliverables = project_root / "SPRINT2_DELIVERABLES" / "figures"
print(f"\n📦 Copying to deliverables...")
for png in png_files:
    import shutil
    shutil.copy(png, deliverables / png.name)
    print(f"   ✅ {png.name}")

print("\n💡 Utilisation dans LaTeX:")
print("   \\includegraphics[width=0.85\\textwidth]{SPRINT2_DELIVERABLES/figures/test1_shock_motos.png}")
