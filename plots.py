import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from pathlib import Path

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
    
    # Fit in log space for better results
    log_N = np.log(N_fit)
    log_R2 = np.log(R2_fit)
    coeffs = np.polyfit(log_N, log_R2, 1)
    nu = coeffs[0] / 2  # Since R^2 ~ N^(2*nu)
    
    return nu, N_fit, power_law(N_fit, np.exp(coeffs[1]), 2*nu)

def plot_ideal_polymer():
    """Plot results for ideal polymer"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
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
                    label=f'{label}: ν = {nu_P:.3f}', markersize=4, alpha=0.7)
        axes[0].plot(N_fit_P, fit_P, '--', color=colors[idx*2], linewidth=2, alpha=0.8)
        
        # Plot R^2_g
        axes[1].plot(N, R2_g, 's', color=colors[idx*2+1], 
                    label=f'{label}: ν = {nu_g:.3f}', markersize=4, alpha=0.7)
        axes[1].plot(N_fit_g, fit_g, '--', color=colors[idx*2+1], linewidth=2, alpha=0.8)
    
    # Configure axes
    for ax, title in zip(axes, ['End-to-End Distance', 'Radius of Gyration']):
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of Monomers (N)', fontsize=11, fontweight='bold')
        ax.set_ylabel('$\\langle R^2 \\rangle$', fontsize=11, fontweight='bold')
        ax.set_title(f'Ideal Polymer: {title}', fontsize=12, fontweight='bold')
        ax.legend(frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ideal_polymer_results.png', bbox_inches='tight')
    print("Saved: ideal_polymer_results.png")
    plt.close()

def plot_excluded_volume_polymer():
    """Plot results for excluded volume polymer"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
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
               label=f'Bare: ν = {nu_bare:.3f}', markersize=5, alpha=0.6)
        ax.plot(N_fit, fit_bare, '-', color=colors[dim_idx*2], linewidth=2.5, alpha=0.8)
        
        # Rosenbluth average
        N_rosen = df_rosen['N'].values
        R2_P_rosen = df_rosen['R2_P'].values
        nu_rosen, N_fit, fit_rosen = fit_power_law(N_rosen, R2_P_rosen, N_min=5)
        
        ax.plot(N_rosen, R2_P_rosen, 's', color=colors[dim_idx*2+1], 
               label=f'Rosenbluth: ν = {nu_rosen:.3f}', markersize=5, alpha=0.6)
        ax.plot(N_fit, fit_rosen, '--', color=colors[dim_idx*2+1], linewidth=2.5, alpha=0.8)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of Monomers (N)', fontsize=10, fontweight='bold')
        ax.set_ylabel('$\\langle R_P^2 \\rangle$', fontsize=10, fontweight='bold')
        ax.set_title(f'{dim_label}: End-to-End Distance', fontsize=11, fontweight='bold')
        ax.legend(frameon=True, shadow=True, loc='best')
        ax.grid(True, alpha=0.3)
        
        # R^2_g comparison
        ax = axes[dim_idx, 1]
        
        # Bare average
        R2_g_bare = df_bare['R2_g'].values
        nu_g_bare, N_fit, fit_g_bare = fit_power_law(N_bare, R2_g_bare, N_min=5)
        
        ax.plot(N_bare, R2_g_bare, 'o', color=colors[dim_idx*2+4], 
               label=f'Bare: ν = {nu_g_bare:.3f}', markersize=5, alpha=0.6)
        ax.plot(N_fit, fit_g_bare, '-', color=colors[dim_idx*2+4], linewidth=2.5, alpha=0.8)
        
        # Rosenbluth average
        R2_g_rosen = df_rosen['R2_g'].values
        nu_g_rosen, N_fit, fit_g_rosen = fit_power_law(N_rosen, R2_g_rosen, N_min=5)
        
        ax.plot(N_rosen, R2_g_rosen, 's', color=colors[dim_idx*2+5], 
               label=f'Rosenbluth: ν = {nu_g_rosen:.3f}', markersize=5, alpha=0.6)
        ax.plot(N_fit, fit_g_rosen, '--', color=colors[dim_idx*2+5], linewidth=2.5, alpha=0.8)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of Monomers (N)', fontsize=10, fontweight='bold')
        ax.set_ylabel('$\\langle R_g^2 \\rangle$', fontsize=10, fontweight='bold')
        ax.set_title(f'{dim_label}: Radius of Gyration', fontsize=11, fontweight='bold')
        ax.legend(frameon=True, shadow=True, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Excluded Volume Polymer: Self-Avoiding Walk', 
                fontsize=14, fontweight='bold', y=0.995)
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
            
            results.append(['Ideal', f'{dim}D', 'R²_P', f'{nu_P:.4f}'])
            results.append(['Ideal', f'{dim}D', 'R²_g', f'{nu_g:.4f}'])
    
    # Excluded volume - Rosenbluth method
    for dim in [2, 3]:
        filename = f'excluded_{dim}d_rosenbluth.csv'
        if Path(filename).exists():
            df = pd.read_csv(filename)
            N = df['N'].values
            R2_P = df['R2_P'].values
            R2_g = df['R2_g'].values
            
            nu_P, _, _ = fit_power_law(N, R2_P, N_min=5)
            nu_g, _, _ = fit_power_law(N, R2_g, N_min=5)
            
            results.append(['Excluded Volume', f'{dim}D', 'R²_P', f'{nu_P:.4f}'])
            results.append(['Excluded Volume', f'{dim}D', 'R²_g', f'{nu_g:.4f}'])
    
    # Create DataFrame
    df_summary = pd.DataFrame(results, columns=['Model', 'Dimension', 'Quantity', 'ν'])
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Pivot for better display
    pivot = df_summary.pivot_table(index=['Model', 'Dimension'], 
                                   columns='Quantity', 
                                   values='ν', 
                                   aggfunc='first')
    
    table = ax.table(cellText=pivot.values,
                    rowLabels=[f'{r[0]} ({r[1]})' for r in pivot.index],
                    colLabels=pivot.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(pivot.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(len(pivot.index)):
        table[(i+1, -1)].set_facecolor('#f0f0f0')
        table[(i+1, -1)].set_text_props(weight='bold')
    
    ax.set_title('Summary: Scaling Exponents ν', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add theoretical values as text
    theory_text = ("Theoretical values:\n"
                  "• Ideal chain: ν = 0.5\n"
                  "• SAW (2D): ν ≈ 0.75\n"
                  "• SAW (3D): ν ≈ 0.588")
    ax.text(0.5, -0.15, theory_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('exponents_summary.png', bbox_inches='tight')
    print("Saved: exponents_summary.png")
    plt.close()
    
    # Also print to console
    print("\n" + "="*60)
    print("SUMMARY: SCALING EXPONENTS ν")
    print("="*60)
    print(df_summary.to_string(index=False))
    print("\nTheoretical values:")
    print("  Ideal chain: ν = 0.5")
    print("  SAW (2D): ν ≈ 0.75")
    print("  SAW (3D): ν ≈ 0.588")
    print("="*60)

def main():
    print("Generating visualizations...")
    print("-" * 60)
    
    plot_ideal_polymer()
    plot_excluded_volume_polymer()
    create_summary_table()
    
    print("-" * 60)
    print("All visualizations complete!")

if __name__ == "__main__":
    main()