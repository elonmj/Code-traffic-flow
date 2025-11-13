"""
Script to generate all figures for the mathematical presentation.
Creates professional scientific visualizations for the numerical methods.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.patches import Wedge
import matplotlib.patches as mpatches

# Set global style for professional scientific figures
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

def create_traffic_jam_figure():
    """Create a conceptual traffic jam visualization"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Road with multiple lanes
    road_y = [0, 1, 2, 3]
    x = np.linspace(0, 10, 100)
    
    # Background - road
    for y in road_y:
        ax.fill_between(x, y-0.4, y+0.4, color='gray', alpha=0.3)
        ax.plot(x, [y]*len(x), 'w--', linewidth=1, alpha=0.5)
    
    # Traffic density visualization using color gradient
    # Dense traffic on left (jam), free flow on right
    for i, xi in enumerate(np.linspace(0, 10, 50)):
        density = np.exp(-(xi-2)**2/2)  # Peak at x=2
        for y in road_y:
            color_intensity = 0.2 + 0.8 * density
            ax.plot([xi, xi], [y-0.35, y+0.35], 
                   color=(color_intensity, 0.1, 0.1), 
                   linewidth=6, alpha=0.8)
    
    # Add vehicles representation
    np.random.seed(42)
    for y in road_y:
        # Dense area
        for _ in range(15):
            x_pos = np.random.uniform(0.5, 3.5)
            car_width = 0.15
            car_height = 0.3
            rect = Rectangle((x_pos, y-car_height/2), car_width, car_height,
                           facecolor='darkred', edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
        
        # Medium density area
        for _ in range(8):
            x_pos = np.random.uniform(3.5, 6)
            car_width = 0.15
            car_height = 0.3
            rect = Rectangle((x_pos, y-car_height/2), car_width, car_height,
                           facecolor='orange', edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
        
        # Free flow area
        for _ in range(4):
            x_pos = np.random.uniform(6, 9.5)
            car_width = 0.15
            car_height = 0.3
            rect = Rectangle((x_pos, y-car_height/2), car_width, car_height,
                           facecolor='green', edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
    
    # Labels
    ax.text(2, -0.8, 'Zone congestionnée\n(densité élevée)', 
           ha='center', fontsize=12, weight='bold', color='darkred')
    ax.text(7.5, -0.8, 'Écoulement fluide\n(densité faible)', 
           ha='center', fontsize=12, weight='bold', color='green')
    
    ax.arrow(4.5, -1.5, 3, 0, head_width=0.3, head_length=0.3, 
            fc='blue', ec='blue', linewidth=2)
    ax.text(6, -2.2, 'Direction du trafic', ha='center', fontsize=11, color='blue')
    
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-2.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Formation d\'un embouteillage : transition de phase', 
                fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('traffic_jam.jpg', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Created traffic_jam.jpg")
    plt.close()


def create_fluid_flow_figure():
    """Create a fluid flow analogy visualization"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create streamlines representing traffic flow
    Y, X = np.mgrid[0:4:100j, 0:10:200j]
    
    # Velocity field (slowing down in the middle)
    U = np.ones_like(X) * (1 - 0.7 * np.exp(-((X-5)**2 + (Y-2)**2)/3))
    V = np.zeros_like(Y)
    
    # Density field (high in the middle)
    density = np.exp(-((X-5)**2 + (Y-2)**2)/4)
    
    # Plot density as contour
    contour = ax.contourf(X, Y, density, levels=20, cmap='YlOrRd', alpha=0.6)
    cbar = plt.colorbar(contour, ax=ax, label='Densité ρ(x,t)')
    
    # Plot streamlines
    ax.streamplot(X, Y, U, V, color='blue', linewidth=1.5, 
                 density=1.5, arrowsize=1.5, arrowstyle='->')
    
    # Add velocity vectors at specific points
    x_samples = np.linspace(1, 9, 9)
    y_samples = np.linspace(0.5, 3.5, 4)
    for xs in x_samples:
        for ys in y_samples:
            u_val = 1 - 0.7 * np.exp(-((xs-5)**2 + (ys-2)**2)/3)
            ax.arrow(xs, ys, u_val*0.5, 0, head_width=0.15, head_length=0.1,
                    fc='darkblue', ec='darkblue', alpha=0.5, linewidth=1)
    
    # Annotations
    ax.text(5, 2, 'Zone dense\n(vitesse réduite)', ha='center', va='center',
           fontsize=11, weight='bold', 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.text(1.5, 3.2, 'Vitesse v(x,t)', fontsize=11, color='darkblue', weight='bold')
    ax.arrow(1.5, 3, 0.8, 0, head_width=0.1, head_length=0.15,
            fc='darkblue', ec='darkblue', linewidth=2)
    
    ax.set_xlabel('Position spatiale x (km)', fontsize=12)
    ax.set_ylabel('Voies', fontsize=12)
    ax.set_title('Analogie fluide : densité et vitesse du trafic', 
                fontsize=14, weight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('fluid_flow.jpg', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Created fluid_flow.jpg")
    plt.close()


def create_grid_figure():
    """Create a spatial discretization grid visualization"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Grid parameters
    n_cells = 7
    x_start = 0
    x_end = 10
    dx = (x_end - x_start) / n_cells
    
    # Draw cells
    for i in range(n_cells):
        x_left = x_start + i * dx
        x_center = x_left + dx/2
        
        # Cell rectangle
        color = 'lightblue' if i % 2 == 0 else 'lightcyan'
        rect = Rectangle((x_left, 0), dx, 2, 
                        facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Cell center
        ax.plot(x_center, 1, 'ro', markersize=10, zorder=10)
        
        # Cell index
        ax.text(x_center, 1.5, f'$C_{{{i}}}$', ha='center', va='center',
               fontsize=13, weight='bold')
        
        # Cell average value
        ax.text(x_center, 0.5, f'$U_{{{i}}}^n$', ha='center', va='center',
               fontsize=11, color='darkred')
        
        # Interface markers
        if i < n_cells - 1:
            ax.plot([x_left + dx, x_left + dx], [0, 2], 'k-', linewidth=3)
            ax.text(x_left + dx, -0.3, f'$i={i}+\\frac{{1}}{{2}}$', 
                   ha='center', fontsize=10, style='italic')
    
    # First and last interfaces
    ax.plot([x_start, x_start], [0, 2], 'k-', linewidth=3)
    ax.plot([x_end, x_end], [0, 2], 'k-', linewidth=3)
    
    # Delta x annotation
    ax.annotate('', xy=(x_start + 2*dx, -0.8), xytext=(x_start + dx, -0.8),
                arrowprops=dict(arrowstyle='<->', lw=2, color='blue'))
    ax.text(x_start + 1.5*dx, -1.1, '$\\Delta x$', ha='center', 
           fontsize=13, color='blue', weight='bold')
    
    # Spatial axis
    ax.arrow(x_start-0.5, -1.8, x_end-x_start+0.7, 0, 
            head_width=0.2, head_length=0.2, fc='black', ec='black', linewidth=1.5)
    ax.text(x_end+0.5, -1.8, '$x$', fontsize=14, weight='bold')
    
    # Title and labels
    ax.text(x_start + (x_end-x_start)/2, 2.8, 
           'Discrétisation Spatiale : Grille de Volumes Finis',
           ha='center', fontsize=14, weight='bold')
    ax.text(x_start + (x_end-x_start)/2, 2.4, 
           'Chaque cellule stocke une valeur moyenne',
           ha='center', fontsize=11, style='italic')
    
    ax.set_xlim(x_start-1, x_end+1)
    ax.set_ylim(-2, 3.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('grid.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Created grid.png")
    plt.close()


def create_reconstruction_figure():
    """Create WENO reconstruction visualization"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Cell centers and values
    x_centers = np.array([0, 1, 2, 3, 4, 5, 6])
    cell_values = np.array([1.0, 1.2, 2.5, 2.8, 2.9, 2.7, 1.5])
    dx = 1.0
    
    # Fine grid for reconstruction
    x_fine = np.linspace(-0.5, 6.5, 500)
    
    # ============= TOP: Piecewise constant (bad) =============
    ax1.set_title('Reconstruction d\'ordre 0 (constante par morceaux) - Diffusive',
                 fontsize=13, weight='bold')
    
    # True solution (smooth)
    def true_solution(x):
        return 1.0 + 1.5*np.exp(-((x-2.5)**2)/2) + 0.3*np.sin(x)
    
    ax1.plot(x_fine, true_solution(x_fine), 'g-', linewidth=3, 
            label='Solution exacte', alpha=0.7)
    
    # Piecewise constant reconstruction
    for i, (xc, val) in enumerate(zip(x_centers, cell_values)):
        x_left = xc - dx/2
        x_right = xc + dx/2
        ax1.plot([x_left, x_right], [val, val], 'b-', linewidth=2.5)
        if i < len(x_centers) - 1:
            ax1.plot([x_right, x_right], [val, cell_values[i+1]], 'b--', 
                    linewidth=1.5, alpha=0.5)
        ax1.plot(xc, val, 'ro', markersize=8, label='Moyenne cellule' if i == 0 else '')
    
    ax1.axvline(3.5, color='red', linestyle=':', linewidth=2, alpha=0.7)
    ax1.text(3.5, 3.5, 'Interface', ha='center', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax1.text(1, 3.5, 'PROBLÈME: Perte d\'information\n(diffusion numérique)', 
            fontsize=11, weight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('Position x', fontsize=11)
    ax1.set_ylabel('Valeur U', fontsize=11)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, 6.5)
    ax1.set_ylim(0.5, 4)
    
    # ============= BOTTOM: WENO5 (good) =============
    ax2.set_title('Reconstruction WENO5 (ordre 5) - Précise et stable',
                 fontsize=13, weight='bold')
    
    ax2.plot(x_fine, true_solution(x_fine), 'g-', linewidth=3, 
            label='Solution exacte', alpha=0.7)
    
    # Simulate WENO reconstruction (smooth polynomial interpolation)
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(x_centers, cell_values, bc_type='natural')
    weno_reconstruction = cs(x_fine)
    
    ax2.plot(x_fine, weno_reconstruction, 'b-', linewidth=2.5, 
            label='Reconstruction WENO5')
    
    for i, (xc, val) in enumerate(zip(x_centers, cell_values)):
        ax2.plot(xc, val, 'ro', markersize=8, label='Moyenne cellule' if i == 0 else '')
    
    # Show stencils
    target_cell = 3
    stencil_cells = [1, 2, 3, 4, 5]
    for sc in stencil_cells:
        ax2.axvspan(sc-dx/2, sc+dx/2, alpha=0.1, color='orange')
    ax2.text(3, 0.7, 'Stencil WENO\n(5 cellules)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
    
    ax2.axvline(3.5, color='red', linestyle=':', linewidth=2, alpha=0.7)
    ax2.text(3.5, 3.5, 'Interface\n(valeur précise)', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax2.text(5.5, 3.5, 'WENO: Haute précision\n+ Stabilité aux chocs', 
            fontsize=11, weight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Position x', fontsize=11)
    ax2.set_ylabel('Valeur U', fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, 6.5)
    ax2.set_ylim(0.5, 4)
    
    plt.tight_layout()
    plt.savefig('reconstruction.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Created reconstruction.png")
    plt.close()


def create_riemann_figure():
    """Create Riemann problem visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ============= LEFT: Initial condition =============
    x = np.linspace(-2, 2, 1000)
    
    # Riemann initial data
    U_L = 2.0
    U_R = 0.5
    
    U_init = np.where(x < 0, U_L, U_R)
    
    ax1.plot(x, U_init, 'b-', linewidth=3, label='Condition initiale')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Annotations
    ax1.plot([-1], [U_L], 'go', markersize=15, label='$U_L$ (gauche)')
    ax1.plot([1], [U_R], 'mo', markersize=15, label='$U_R$ (droite)')
    
    ax1.text(-1, U_L+0.3, f'$U_L = {U_L}$', ha='center', fontsize=13, 
            weight='bold', color='green')
    ax1.text(1, U_R-0.3, f'$U_R = {U_R}$', ha='center', fontsize=13, 
            weight='bold', color='magenta')
    
    ax1.text(0, 2.8, 'Interface\n$x_{i+1/2}$', ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax1.set_xlabel('Position x', fontsize=12)
    ax1.set_ylabel('État U', fontsize=12)
    ax1.set_title('Problème de Riemann : État Initial (t=0)', fontsize=13, weight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 3)
    ax1.set_xlim(-2, 2)
    
    # ============= RIGHT: Solution structure =============
    t = 0.5
    
    # Wave speeds (characteristic speeds)
    a_L = -1.5  # Left-going wave
    a_R = 1.8   # Right-going wave
    U_star = 1.2  # Intermediate state
    
    # Solution structure
    x_left_wave = a_L * t
    x_right_wave = a_R * t
    
    U_solution = np.piecewise(x, 
                              [x < x_left_wave, 
                               (x >= x_left_wave) & (x < x_right_wave),
                               x >= x_right_wave],
                              [U_L, U_star, U_R])
    
    ax2.plot(x, U_solution, 'b-', linewidth=3, label='Solution à t>0')
    ax2.axvline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Mark wave positions
    ax2.axvline(x_left_wave, color='purple', linestyle='--', linewidth=2, 
               label=f'Onde gauche (a_L={a_L})')
    ax2.axvline(x_right_wave, color='orange', linestyle='--', linewidth=2,
               label=f'Onde droite (a_R={a_R})')
    
    # Regions
    ax2.fill_between([x_left_wave, x_right_wave], 0, 3, alpha=0.2, color='cyan',
                    label='Zone de contact')
    
    ax2.text(x_left_wave, 2.8, '$a_L \\cdot t$', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='purple', alpha=0.5))
    ax2.text(x_right_wave, 2.8, '$a_R \\cdot t$', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
    ax2.text(0, 1.8, '$U^*$\n(état\nintermédiaire)', ha='center', fontsize=11,
            weight='bold', bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.7))
    
    # Flux annotation
    ax2.annotate('Flux $F_{i+1/2}$\ncalculé ici', xy=(0, 1.2), xytext=(0.8, 0.3),
                fontsize=11, weight='bold', color='red',
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax2.set_xlabel('Position x', fontsize=12)
    ax2.set_ylabel('État U', fontsize=12)
    ax2.set_title(f'Solution du Problème de Riemann (t={t})', 
                 fontsize=13, weight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 3)
    ax2.set_xlim(-2, 2)
    
    plt.tight_layout()
    plt.savefig('riemann.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Created riemann.png")
    plt.close()


def main():
    """Create all figures for the presentation"""
    print("\n" + "="*60)
    print("Creating all figures for the mathematical presentation...")
    print("="*60 + "\n")
    
    try:
        create_traffic_jam_figure()
        create_fluid_flow_figure()
        create_grid_figure()
        create_reconstruction_figure()
        create_riemann_figure()
        
        print("\n" + "="*60)
        print("✓ ALL FIGURES CREATED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        print("  - traffic_jam.jpg")
        print("  - fluid_flow.jpg")
        print("  - grid.png")
        print("  - reconstruction.png")
        print("  - riemann.png")
        print("\nYou can now compile the LaTeX presentation.")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
