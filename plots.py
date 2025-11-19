import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

# Set seaborn style for professional plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

def power_law(x, a, nu):
    """Power law: y = a * x^nu"""
    return a * x**nu

def fit_power_law(N, R2, N_min=10):
    """Fit power law and return exponent nu"""
    mask = N >= N_min
    N_fit = N[mask]
    R2_fit = R2[mask]
    
    if len(N_fit) < 3:
        return 0.5, N_fit, R2_fit
    
    # Fit in log space for better results
    log_N = np.log(N_fit)
    log_R2 = np.log(R2_fit)
    coeffs = np.polyfit(log_N, log_R2, 1)
    nu = coeffs[0] / 2  # Since R^2 ~ N^(2*nu)
    
    return nu, N_fit, power_law(N_fit, np.exp(coeffs[1]), 2*nu)

def generate_sample_ideal_2d(N=50, seed=42):
    """Generate a sample ideal polymer (random walk) in 2D"""
    np.random.seed(seed)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    positions = [(0, 0), (1, 0)]
    for _ in range(N - 2):
        last = positions[-1]
        dx, dy = directions[np.random.randint(0, 4)]
        positions.append((last[0] + dx, last[1] + dy))
    
    return np.array(positions)

def generate_sample_saw_2d(N=50, seed=42, max_attempts=100):
    """Generate a sample self-avoiding walk in 2D"""
    np.random.seed(seed)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for attempt in range(max_attempts):
        positions = [(0, 0), (1, 0)]
        occupied = {(0, 0), (1, 0)}
        
        for _ in range(N - 2):
            last = positions[-1]
            available = []
            
            for dx, dy in directions:
                candidate = (last[0] + dx, last[1] + dy)
                if candidate not in occupied:
                    available.append(candidate)
            
            if not available:
                break
            
            new_pos = available[np.random.randint(0, len(available))]
            positions.append(new_pos)
            occupied.add(new_pos)
        
        if len(positions) >= N:
            return np.array(positions)
    
    return np.array(positions)

def generate_sample_ideal_3d(N=50, seed=42):
    """Generate a sample ideal polymer (random walk) in 3D"""
    np.random.seed(seed)
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    
    positions = [(0, 0, 0), (1, 0, 0)]
    for _ in range(N - 2):
        last = positions[-1]
        dx, dy, dz = directions[np.random.randint(0, 6)]
        positions.append((last[0] + dx, last[1] + dy, last[2] + dz))
    
    return np.array(positions)

def generate_sample_saw_3d(N=50, seed=42, max_attempts=100):
    """Generate a sample self-avoiding walk in 3D"""
    np.random.seed(seed)
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    
    for attempt in range(max_attempts):
        positions = [(0, 0, 0), (1, 0, 0)]
        occupied = {(0, 0, 0), (1, 0, 0)}
        
        for _ in range(N - 2):
            last = positions[-1]
            available = []
            
            for dx, dy, dz in directions:
                candidate = (last[0] + dx, last[1] + dy, last[2] + dz)
                if candidate not in occupied:
                    available.append(candidate)
            
            if not available:
                break
            
            new_pos = available[np.random.randint(0, len(available))]
            positions.append(new_pos)
            occupied.add(new_pos)
        
        if len(positions) >= N:
            return np.array(positions)
    
    return np.array(positions)

def plot_polymer_examples_2d():
    """Create side-by-side comparison of ideal and SAW polymers in 2D"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # Generate examples with different lengths
    N_values = [30, 80]
    
    for idx, N in enumerate(N_values):
        # Ideal polymer
        ax = axes[idx, 0]
        ideal = generate_sample_ideal_2d(N, seed=42 + idx)
        
        # Plot bonds
        ax.plot(ideal[:, 0], ideal[:, 1], 'b-', linewidth=1.5, alpha=0.6, label='Bonds')
        
        # Plot monomers with color gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(ideal)))
        for i, (x, y) in enumerate(ideal):
            ax.scatter(x, y, c=[colors[i]], s=80, edgecolors='black', 
                      linewidths=0.5, zorder=5)
        
        # Mark start and end
        ax.scatter(ideal[0, 0], ideal[0, 1], c='red', s=200, marker='*', 
                  edgecolors='black', linewidths=1, zorder=10, label='Start')
        ax.scatter(ideal[-1, 0], ideal[-1, 1], c='gold', s=200, marker='*', 
                  edgecolors='black', linewidths=1, zorder=10, label='End')
        
        # Calculate metrics
        R_squared = (ideal[-1, 0] - ideal[0, 0])**2 + (ideal[-1, 1] - ideal[0, 1])**2
        cm = ideal.mean(axis=0)
        Rg_squared = np.mean(np.sum((ideal - cm)**2, axis=1))
        
        ax.set_xlabel('x', fontsize=11, fontweight='bold')
        ax.set_ylabel('y', fontsize=11, fontweight='bold')
        ax.set_title(f'Ideal Polymer (Random Walk)\nN={N}, R²={R_squared:.1f}, R²g={Rg_squared:.1f}', 
                    fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # SAW polymer
        ax = axes[idx, 1]
        saw = generate_sample_saw_2d(N, seed=42 + idx)
        
        # Plot bonds
        ax.plot(saw[:, 0], saw[:, 1], 'g-', linewidth=1.5, alpha=0.6, label='Bonds')
        
        # Plot monomers with color gradient
        colors = plt.cm.plasma(np.linspace(0, 1, len(saw)))
        for i, (x, y) in enumerate(saw):
            ax.scatter(x, y, c=[colors[i]], s=80, edgecolors='black', 
                      linewidths=0.5, zorder=5)
        
        # Mark start and end
        ax.scatter(saw[0, 0], saw[0, 1], c='red', s=200, marker='*', 
                  edgecolors='black', linewidths=1, zorder=10, label='Start')
        ax.scatter(saw[-1, 0], saw[-1, 1], c='gold', s=200, marker='*', 
                  edgecolors='black', linewidths=1, zorder=10, label='End')
        
        # Calculate metrics
        R_squared = (saw[-1, 0] - saw[0, 0])**2 + (saw[-1, 1] - saw[0, 1])**2
        cm = saw.mean(axis=0)
        Rg_squared = np.mean(np.sum((saw - cm)**2, axis=1))
        
        ax.set_xlabel('x', fontsize=11, fontweight='bold')
        ax.set_ylabel('y', fontsize=11, fontweight='bold')
        ax.set_title(f'Self-Avoiding Walk\nN={len(saw)}, R²={R_squared:.1f}, R²g={Rg_squared:.1f}', 
                    fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.suptitle('Polymer Conformations in 2D: Ideal vs. Self-Avoiding Walk', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('polymer_examples_2d.png', bbox_inches='tight')
    print("Saved: polymer_examples_2d.png")
    plt.close()

def plot_polymer_examples_3d():
    """Create side-by-side comparison of ideal and SAW polymers in 3D"""
    fig = plt.figure(figsize=(16, 8))
    
    N = 60
    
    # Ideal polymer 3D
    ax1 = fig.add_subplot(121, projection='3d')
    ideal = generate_sample_ideal_3d(N, seed=42)
    
    # Plot bonds
    ax1.plot(ideal[:, 0], ideal[:, 1], ideal[:, 2], 'b-', linewidth=1.5, alpha=0.5)
    
    # Plot monomers with color gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(ideal)))
    for i, (x, y, z) in enumerate(ideal):
        ax1.scatter(x, y, z, c=[colors[i]], s=50, edgecolors='black', 
                   linewidths=0.5, alpha=0.8)
    
    # Mark start and end
    ax1.scatter(ideal[0, 0], ideal[0, 1], ideal[0, 2], c='red', s=150, 
               marker='*', edgecolors='black', linewidths=1)
    ax1.scatter(ideal[-1, 0], ideal[-1, 1], ideal[-1, 2], c='gold', s=150, 
               marker='*', edgecolors='black', linewidths=1)
    
    R_squared = np.sum((ideal[-1] - ideal[0])**2)
    cm = ideal.mean(axis=0)
    Rg_squared = np.mean(np.sum((ideal - cm)**2, axis=1))
    
    ax1.set_xlabel('x', fontsize=10, fontweight='bold')
    ax1.set_ylabel('y', fontsize=10, fontweight='bold')
    ax1.set_zlabel('z', fontsize=10, fontweight='bold')
    ax1.set_title(f'Ideal Polymer (3D Random Walk)\nN={N}, R²={R_squared:.1f}, R²g={Rg_squared:.1f}', 
                 fontsize=11, fontweight='bold', pad=20)
    
    # SAW polymer 3D
    ax2 = fig.add_subplot(122, projection='3d')
    saw = generate_sample_saw_3d(N, seed=42)
    
    # Plot bonds
    ax2.plot(saw[:, 0], saw[:, 1], saw[:, 2], 'g-', linewidth=1.5, alpha=0.5)
    
    # Plot monomers with color gradient
    colors = plt.cm.plasma(np.linspace(0, 1, len(saw)))
    for i, (x, y, z) in enumerate(saw):
        ax2.scatter(x, y, z, c=[colors[i]], s=50, edgecolors='black', 
                   linewidths=0.5, alpha=0.8)
    
    # Mark start and end
    ax2.scatter(saw[0, 0], saw[0, 1], saw[0, 2], c='red', s=150, 
               marker='*', edgecolors='black', linewidths=1)
    ax2.scatter(saw[-1, 0], saw[-1, 1], saw[-1, 2], c='gold', s=150, 
               marker='*', edgecolors='black', linewidths=1)
    
    R_squared = np.sum((saw[-1] - saw[0])**2)
    cm = saw.mean(axis=0)
    Rg_squared = np.mean(np.sum((saw - cm)**2, axis=1))
    
    ax2.set_xlabel('x', fontsize=10, fontweight='bold')
    ax2.set_ylabel('y', fontsize=10, fontweight='bold')
    ax2.set_zlabel('z', fontsize=10, fontweight='bold')
    ax2.set_title(f'Self-Avoiding Walk (3D)\nN={len(saw)}, R²={R_squared:.1f}, R²g={Rg_squared:.1f}', 
                 fontsize=11, fontweight='bold', pad=20)
    
    plt.suptitle('Polymer Conformations in 3D', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('polymer_examples_3d.png', bbox_inches='tight')
    print("Saved: polymer_examples_3d.png")
    plt.close()

def plot_ideal_polymer():
    """Plot results for ideal polymer"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = sns.color_palette("husl", 4)
    
    for idx, (dim, label) in enumerate([(2, '2D'), (3, '3D')]):
        filename = f'ideal_{dim}d.csv'
        if not Path(filename).exists():
            print(f"Warning: {filename} not found")
            continue
            
        df = pd.read_csv(filename)
        N = df['N'].values
        R2_P = df['R2_P'].values
        R2_g = df['R2_g'].values
        
        # Fit power laws
        nu_P, N_fit_P, fit_P = fit_power_law(N, R2_P)
        nu_g, N_fit_g, fit_g = fit_power_law(N, R2_g)
        
        # Plot R^2_P
        axes[0].plot(N, R2_P, 'o', color=colors[idx*2], 
                    label=f'{label}: ν = {nu_P:.3f}', markersize=5, alpha=0.7)
        axes[0].plot(N_fit_P, fit_P, '--', color=colors[idx*2], linewidth=2.5, alpha=0.8)
        
        # Plot R^2_g
        axes[1].plot(N, R2_g, 's', color=colors[idx*2+1], 
                    label=f'{label}: ν = {nu_g:.3f}', markersize=5, alpha=0.7)
        axes[1].plot(N_fit_g, fit_g, '--', color=colors[idx*2+1], linewidth=2.5, alpha=0.8)
    
    # Configure axes
    for ax, title in zip(axes, ['End-to-End Distance', 'Radius of Gyration']):
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of Monomers (N)', fontsize=12, fontweight='bold')
        ax.set_ylabel('$\\langle R^2 \\rangle$', fontsize=12, fontweight='bold')
        ax.set_title(f'Ideal Polymer: {title}', fontsize=13, fontweight='bold')
        ax.legend(frameon=True, shadow=True, fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ideal_polymer_results.png', bbox_inches='tight')
    print("Saved: ideal_polymer_results.png")
    plt.close()

def plot_excluded_volume_polymer():
    """Plot results for excluded volume polymer"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    
    colors = sns.color_palette("Set2", 8)
    
    for dim_idx, (dim, dim_label) in enumerate([(2, '2D'), (3, '3D')]):
        bare_file = f'excluded_{dim}d_bare.csv'
        rosen_file = f'excluded_{dim}d_rosenbluth.csv'
        
        if not Path(bare_file).exists() or not Path(rosen_file).exists():
            print(f"Warning: files for {dim}D not found")
            continue
        
        df_bare = pd.read_csv(bare_file)
        df_rosen = pd.read_csv(rosen_file)
        
        # R^2_P comparison
        ax = axes[dim_idx, 0]
        
        # Bare average
        N_bare = df_bare['N'].values
        R2_P_bare = df_bare['R2_P'].values
        nu_bare, N_fit, fit_bare = fit_power_law(N_bare, R2_P_bare, N_min=5)
        
        ax.plot(N_bare, R2_P_bare, 'o', color=colors[dim_idx*2], 
               label=f'Bare: ν = {nu_bare:.3f}', markersize=6, alpha=0.6)
        ax.plot(N_fit, fit_bare, '-', color=colors[dim_idx*2], linewidth=2.5, alpha=0.8)
        
        # Rosenbluth average
        N_rosen = df_rosen['N'].values
        R2_P_rosen = df_rosen['R2_P'].values
        nu_rosen, N_fit, fit_rosen = fit_power_law(N_rosen, R2_P_rosen, N_min=5)
        
        ax.plot(N_rosen, R2_P_rosen, 's', color=colors[dim_idx*2+1], 
               label=f'Rosenbluth: ν = {nu_rosen:.3f}', markersize=6, alpha=0.6)
        ax.plot(N_fit, fit_rosen, '--', color=colors[dim_idx*2+1], linewidth=2.5, alpha=0.8)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of Monomers (N)', fontsize=11, fontweight='bold')
        ax.set_ylabel('$\\langle R_P^2 \\rangle$', fontsize=11, fontweight='bold')
        ax.set_title(f'{dim_label}: End-to-End Distance', fontsize=12, fontweight='bold')
        ax.legend(frameon=True, shadow=True, loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # R^2_g comparison
        ax = axes[dim_idx, 1]
        
        # Bare average
        R2_g_bare = df_bare['R2_g'].values
        nu_g_bare, N_fit, fit_g_bare = fit_power_law(N_bare, R2_g_bare, N_min=5)
        
        ax.plot(N_bare, R2_g_bare, 'o', color=colors[dim_idx*2+4], 
               label=f'Bare: ν = {nu_g_bare:.3f}', markersize=6, alpha=0.6)
        ax.plot(N_fit, fit_g_bare, '-', color=colors[dim_idx*2+4], linewidth=2.5, alpha=0.8)
        
        # Rosenbluth average
        R2_g_rosen = df_rosen['R2_g'].values
        nu_g_rosen, N_fit, fit_g_rosen = fit_power_law(N_rosen, R2_g_rosen, N_min=5)
        
        ax.plot(N_rosen, R2_g_rosen, 's', color=colors[dim_idx*2+5], 
               label=f'Rosenbluth: ν = {nu_g_rosen:.3f}', markersize=6, alpha=0.6)
        ax.plot(N_fit, fit_g_rosen, '--', color=colors[dim_idx*2+5], linewidth=2.5, alpha=0.8)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of Monomers (N)', fontsize=11, fontweight='bold')
        ax.set_ylabel('$\\langle R_g^2 \\rangle$', fontsize=11, fontweight='bold')
        ax.set_title(f'{dim_label}: Radius of Gyration', fontsize=12, fontweight='bold')
        ax.legend(frameon=True, shadow=True, loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Excluded Volume Polymer: Self-Avoiding Walk', 
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('excluded_volume_polymer_results.png', bbox_inches='tight')
    print("Saved: excluded_volume_polymer_results.png")
    plt.close()

def create_summary_table():
    """Create a summary table of all exponents"""
    results = []
    
    # Ideal polymer
    for dim in [2, 3]:
        filename = f'ideal_{dim}d.csv'
        if Path(filename).exists():
            df = pd.read_csv(filename)
            N = df['N'].values
            R2_P = df['R2_P'].values
            R2_g = df['R2_g'].values
            
            nu_P, _, _ = fit_power_law(N, R2_P)
            nu_g, _, _ = fit_power_law(N, R2_g)
            
            results.append(['Ideal', f'{dim}D', 'R²_P', f'{nu_P:.4f}', '0.5000'])
            results.append(['Ideal', f'{dim}D', 'R²_g', f'{nu_g:.4f}', '0.5000'])
    
    # Excluded volume - Rosenbluth method
    theory_2d = '0.7500'
    theory_3d = '0.5880'
    for dim in [2, 3]:
        filename = f'excluded_{dim}d_rosenbluth.csv'
        if Path(filename).exists():
            df = pd.read_csv(filename)
            N = df['N'].values
            R2_P = df['R2_P'].values
            R2_g = df['R2_g'].values
            
            nu_P, _, _ = fit_power_law(N, R2_P, N_min=5)
            nu_g, _, _ = fit_power_law(N, R2_g, N_min=5)
            
            theory = theory_2d if dim == 2 else theory_3d
            results.append(['Excluded Volume', f'{dim}D', 'R²_P', f'{nu_P:.4f}', theory])
            results.append(['Excluded Volume', f'{dim}D', 'R²_g', f'{nu_g:.4f}', theory])
    
    # Create DataFrame
    df_summary = pd.DataFrame(results, columns=['Model', 'Dimension', 'Quantity', 'ν (simulated)', 'ν (theory)'])
    
    # Print to console
    print("\n" + "="*80)
    print("SUMMARY: SCALING EXPONENTS ν")
    print("="*80)
    print(df_summary.to_string(index=False))
    print("\nTheoretical values:")
    print("  Ideal chain: ν = 0.5")
    print("  SAW (2D): ν ≈ 0.75")
    print("  SAW (3D): ν ≈ 0.588")
    print("="*80)
    
    return df_summary

def main():
    print("Generating enhanced visualizations with monomer representations...")
    print("-" * 60)
    
    # Generate monomer visualizations
    print("\n1. Creating 2D polymer examples with monomers...")
    plot_polymer_examples_2d()
    
    print("\n2. Creating 3D polymer examples with monomers...")
    plot_polymer_examples_3d()
    
    # Generate analysis plots
    print("\n3. Analyzing ideal polymer data...")
    plot_ideal_polymer()
    
    print("\n4. Analyzing excluded volume polymer data...")
    plot_excluded_volume_polymer()
    
    print("\n5. Creating summary table...")
    summary_df = create_summary_table()
    
    print("\n" + "-" * 60)
    print("All visualizations complete!")
    print("\nGenerated files:")
    print("  • polymer_examples_2d.png - Monomer visualizations in 2D")
    print("  • polymer_examples_3d.png - Monomer visualizations in 3D")
    print("  • ideal_polymer_results.png - Scaling analysis for ideal polymers")
    print("  • excluded_volume_polymer_results.png - SAW analysis")

if __name__ == "__main__":
    main()