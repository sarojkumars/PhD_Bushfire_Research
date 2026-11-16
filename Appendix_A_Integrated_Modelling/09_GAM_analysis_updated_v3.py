

import pandas as pd
import numpy as np
from pygam import LinearGAM, s, te
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import time
from datetime import datetime
import os
from pathlib import Path
from scipy import stats
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings('ignore')

plt.style.use('default')

print("Starting fire severity GAM analysis...")
start_time = time.time()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "appendix_gam_outputs_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_dir = str(OUTPUT_DIR)  
print(f"\nOutput directory: {OUTPUT_DIR}")

try:
    from appendix_config import DATA_FILE
except ImportError as exc:
    raise ImportError(
        "Could not import DATA_FILE from appendix_config.py. Place this GAM script in "
        "the same directory as appendix_config.py."
    ) from exc

file_path = Path(DATA_FILE)
if not file_path.exists():
    raise FileNotFoundError(f"Fire severity modelling dataset not found: {file_path}")

data = pd.read_csv(file_path)
data['LF4Cls'] = data['LF4Cls'].astype('category')

if 'NDMI_mean' in data.columns:
    NDMI_VAR = 'NDMI_mean'
elif 'NDMI_mean_OctJan' in data.columns:
    NDMI_VAR = 'NDMI_mean_OctJan'
else:
    raise ValueError("Neither 'NDMI_mean' nor legacy 'NDMI_mean_OctJan' was found in DATA_FILE.")

print(f"Data loaded successfully: {file_path}")
print(f"NDMI field used: {NDMI_VAR}")

partial_plot_vars = ['CHM', NDMI_VAR, 'LAI', 'DFFM_silo', 'DEM', 'SlpPerc', 'AspCosine', 'DwnSlpCurv', 'UpSlpCurv','SoilMois_0to10']

interaction_pairs = [
    ('CHM', 'LAI'),
    ('DFFM_silo', 'SlpPerc'),
    (NDMI_VAR, 'SlpPerc'),
    ('SoilMois_0to10', 'AspCosine'),
    ('SoilMois_0to10', 'DEM'),
    ('CHM', NDMI_VAR),
    ('LAI', NDMI_VAR),
    ('LAI', 'UpSlpCurv'),
    ('LAI', 'DwnSlpCurv'),
    ('AspCosine', 'SlpPerc'),
    ('CHM', 'SlpPerc'),
    ('CHM', 'AspCosine'),
    ('CHM', 'DEM'),
    ('CHM', 'DwnSlpCurv'),
    ('CHM', 'UpSlpCurv')
]

ch_3d_interactions = [
    ('CHM', 'SlpPerc'),
    ('CHM', 'AspCosine'),
    ('CHM', 'DEM'),
    ('CHM', 'DwnSlpCurv'),
    ('CHM', 'UpSlpCurv')
]

all_vars = ['CHM', NDMI_VAR, 'LAI', 'DFFM_silo', 'SoilMois_0to10', 'DEM', 'SlpPerc', 'AspCosine', 'DwnSlpCurv', 'UpSlpCurv']
response_var = 'RdNBR'

var_names = {
    'CHM': 'CH',
    NDMI_VAR: 'NDMI',
    'LAI': 'LAI', 
    'DFFM_silo': 'DFFM',
    'DEM': 'Elev',
    'SlpPerc': 'Slp',
    'AspCosine': 'Asp',
    'DwnSlpCurv': 'DSlpCurv',
    'UpSlpCurv': 'USlpCurv',
    'SoilMois_0to10': 'SMois'
}

def fit_gam_model(data, response_var, predictors, k=20):
    """
    Fit GAM model with smooth terms and tensor interactions
    """
    if len(data) < 50:
        print(f"Too few samples ({len(data)}) for reliable model.")
        return None
    try:
        print(f"Fitting GAM with {len(predictors)} predictors and n_splines={k}...")
        
        smooth_terms = []
        for i, pred in enumerate(predictors):
            smooth_terms.append(s(i, n_splines=k))
        
        if 'SlpPerc' in predictors and 'CHM' in predictors:
            slp_idx = predictors.index('SlpPerc')
            chm_idx = predictors.index('CHM')
            smooth_terms.append(te(slp_idx, chm_idx, n_splines=k))
        if 'SlpPerc' in predictors and NDMI_VAR in predictors:
            slp_idx = predictors.index('SlpPerc')
            ndmi_idx = predictors.index(NDMI_VAR)
            smooth_terms.append(te(slp_idx, ndmi_idx, n_splines=k))
        if 'SlpPerc' in predictors and 'DFFM_silo' in predictors:
            slp_idx = predictors.index('SlpPerc')
            dffm_idx = predictors.index('DFFM_silo')
            smooth_terms.append(te(slp_idx, dffm_idx, n_splines=k))
        
        if len(smooth_terms) == 1:
            gam_terms = smooth_terms[0]
        else:
            gam_terms = smooth_terms[0]
            for term in smooth_terms[1:]:
                gam_terms += term
        
        gam = LinearGAM(gam_terms).fit(data[predictors].values, data[response_var].values)
        print("GAM fitting successful")
        return gam
    except Exception as e:
        print(f"Model fitting error: {e}")
        return None

def calculate_performance_metrics(y_true, y_pred):
    """Calculate RMSE, MAE, and R-squared"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    return rmse, mae, r2

def create_diagnostic_plots(model, y_true, y_pred, residuals, landform, output_dir, timestamp):
    """Create diagnostic plots similar to R's gam.check()"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Histogram of Residuals')
    axes[1, 1].plot(residuals, alpha=0.6)
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Index')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals vs Index')
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f'GAM_diagnostics_{landform}_{timestamp}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diagnostic plots saved: {fig_path}")

def create_2d_interaction_plots(model, data, predictors, landform, output_dir, timestamp):
    """Create 2D interaction plots for specific variable pairs"""

    colors = {
        'line': '#263238',
        'ci68': '#64B5F6',
        'ci95': '#BBDEFB',
        'zero': '#78909C',
        'text': '#263238'
    }

    fig, axes = plt.subplots(3, 5, figsize=(30, 18))
    axes_flat = axes.flatten()
    skipped_pairs = []
    
    for i in range(15):
        ax = axes_flat[i]
        if i >= len(interaction_pairs):
            ax.axis('off')
            continue
        var1, var2 = interaction_pairs[i]
        try:
            if var1 not in predictors or var2 not in predictors:
                msg = f"{var_names.get(var1, var1)} × {var_names.get(var2, var2)}: Variable not in model"
                print(f"[WARNING] Skipping pair: {msg}")
                skipped_pairs.append(msg)
                ax.text(0.5, 0.5, msg, ha='center', va='center', transform=ax.transAxes, fontsize=18, color=colors['text'])
                ax.set_title(f'{var_names.get(var1, var1)} × {var_names.get(var2, var2)}', fontsize=22, fontweight='bold', color=colors['text'], pad=10)
                ax.axis('on')
                continue
            x_range = np.linspace(float(data[var1].min()), float(data[var1].max()), 100)
            y_range = np.linspace(float(data[var2].min()), float(data[var2].max()), 100)
            X, Y = np.meshgrid(x_range, y_range)
            grid_data = np.zeros((X.size, len(predictors)))
            for j, var in enumerate(predictors):
                if var == var1:
                    grid_data[:, j] = X.ravel()
                elif var == var2:
                    grid_data[:, j] = Y.ravel()
                else:
                    grid_data[:, j] = float(data[var].astype(float).median())
            Z = model.predict(grid_data).reshape(X.shape)
            contour = ax.contourf(X, Y, Z, levels=30, cmap='RdYlBu_r', alpha=0.8)
            ax.contour(X, Y, Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)
            x_label = var_names.get(var1, var1)
            y_label = var_names.get(var2, var2)
            plot_label = ["k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y"][i]
            ax.set_title(f'({plot_label}) {x_label} × {y_label}', fontsize=22, fontweight='bold', color='black', pad=10)
            ax.set_xlabel(x_label, fontsize=20, color='black')
            ax.set_ylabel(y_label, fontsize=20, color='black')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=18)
        except Exception as e:
            msg = f"{var_names.get(var1, var1)} × {var_names.get(var2, var2)}: Error: {e}"
            print(f"[ERROR] {msg}")
            skipped_pairs.append(msg)
            ax.text(0.5, 0.5, f'Error plotting\n{var1} × {var2}', ha='center', va='center', transform=ax.transAxes, fontsize=18, color=colors['text'])
            ax.set_title(f'{var_names.get(var1, var1)} × {var_names.get(var2, var2)}', fontsize=22, fontweight='bold', color=colors['text'], pad=10)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    fig_path = os.path.join(output_dir, f'GAM_2d_interactions_{landform}_{timestamp}.png')
    plt.savefig(fig_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"2D Interaction plots saved: {fig_path}")
    if skipped_pairs:
        print("\nSummary of skipped/blank 2D interaction pairs:")
        for msg in skipped_pairs:
            print(" -", msg)

def create_3d_ch_interaction_plots(model, data, predictors, landform, output_dir, timestamp):
    """Create 3D perspective plots for CH interactions in a single row"""

    colors = {
        'primary': '#2E86AB',      
        'secondary': '#A23B72',    
        'accent': '#F18F01',       
        'neutral': '#C73E1D',      
        'background': '#F8F9FA',   
        'text': '#2C3E50'          
    }

    fig, axes = plt.subplots(1, 5, figsize=(30, 6), subplot_kw={'projection': '3d'})

    plot_labels_3d = ["aa", "ab", "ac", "ad", "ae"]

    for i, (var1, var2) in enumerate(ch_3d_interactions):
        if i >= 5:  
            break
        
        ax = axes[i]
        
        try:
            
            if var1 not in predictors or var2 not in predictors:
                ax.text2D(0.5, 0.5, f'{var_names.get(var1, var1)} × {var_names.get(var2, var2)}\nVariable not in model', 
                         ha='center', va='center', transform=ax.transAxes,
                         fontsize=14, color=colors['text'])
                ax.set_title(f'{var_names.get(var1, var1)} × {var_names.get(var2, var2)}', 
                           fontsize=16, fontweight='bold', color=colors['text'], pad=10)
                continue

            if var1 == 'CHM':
                x_min = 0
                x_max = float(data[var1].max())
            else:
                x_min = float(data[var1].min())
                x_max = float(data[var1].max())
            x_range = np.linspace(x_min, x_max, 80)
            y_range = np.linspace(float(data[var2].min()), float(data[var2].max()), 80)
            X, Y = np.meshgrid(x_range, y_range)

            grid_data = np.zeros((X.size, len(predictors)))
            for j, var in enumerate(predictors):
                if var == var1:
                    grid_data[:, j] = X.ravel()
                elif var == var2:
                    grid_data[:, j] = Y.ravel()
                else:
                    grid_data[:, j] = float(data[var].astype(float).median())

            Z = model.predict(grid_data).reshape(X.shape)

            grad_x = np.gradient(Z, axis=1)
            grad_y = np.gradient(Z, axis=0)
            sensitivity = np.sqrt(grad_x**2 + grad_y**2)

            sensitivity_norm = (sensitivity - sensitivity.min()) / (sensitivity.max() - sensitivity.min())

            surf = ax.plot_surface(X, Y, Z,
                                  facecolors=plt.cm.get_cmap('viridis')(sensitivity_norm),
                                  alpha=0.8, linewidth=0, antialiased=True)

            contour = ax.contour(X, Y, Z, 
                                levels=8, colors='black', alpha=0.4, linewidths=0.8)

            norm = plt.Normalize(0, 1)
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.4, aspect=10, pad=0.07, orientation='horizontal')
            cbar.set_label('Sensitivity', fontsize=15.5, color='black')
            cbar.ax.tick_params(labelsize=13.5)
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['Low', 'High'])

            x_label = var_names.get(var1, var1)
            y_label = var_names.get(var2, var2)
            plot_label = plot_labels_3d[i]
            ax.set_title(f'({plot_label}) {x_label} × {y_label}', 
                        fontsize=19, fontweight='bold', color='black', pad=10)
            ax.set_xlabel(x_label, fontsize=17, color='black', labelpad=15)
            ax.set_ylabel(y_label, fontsize=17, color='black', labelpad=15)
            ax.set_zlabel('Effect', fontsize=17, color='black', labelpad=10, rotation=90)

            if var1 == 'CHM':
                ax.set_xlim(0, float(data[var1].max()))

            ax.tick_params(axis='both', which='major', labelsize=13)

            if var2 in ['UpSlpCurv', 'DwnSlpCurv']:
                all_xticks = ax.get_xticks()
                all_yticks = ax.get_yticks()
                ax.set_xticks(all_xticks)
                ax.set_yticks(all_yticks)
                ax.set_xticklabels([f'{x:.0f}' if i % 2 == 0 else '' for i, x in enumerate(all_xticks)])
                ax.set_yticklabels([f'{y:.0f}' if i % 2 == 0 else '' for i, y in enumerate(all_yticks)])
            elif var2 == 'DEM':
                all_xticks = ax.get_xticks()
                ax.set_xticks(all_xticks)
                ax.set_xticklabels([f'{x:.0f}' if i % 2 == 0 else '' for i, x in enumerate(all_xticks)])
            
            ax.view_init(elev=25, azim=45)
            ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            
        except Exception as e:
            print(f"Error plotting 3D CH interaction {var1} × {var2}: {e}")
            ax.text2D(0.5, 0.5, f'Error plotting\n{var1} × {var2}', 
                     ha='center', va='center', transform=ax.transAxes,
                     fontsize=14, color=colors['text'])
            ax.set_title(f'{var_names.get(var1, var1)} × {var_names.get(var2, var2)}', 
                        fontsize=16, fontweight='bold', color=colors['text'], pad=10)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f'GAM_3d_ch_interactions_{landform}_{timestamp}.png')
    plt.savefig(fig_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"3D CH Interaction plots saved: {fig_path}")

def extract_term_edfs(gam, predictors, k):
    """Return a list of (term_name, EDF) for all smooth and tensor terms in the model."""
    edof = gam.statistics_['edof_per_coef']
    term_names = []
    edfs = []
    idx = 0
    
    for pred in predictors:
        term_names.append(f's({pred})')
        edfs.append(float(np.sum(edof[idx:idx+k])))
        idx += k
    
    if 'SlpPerc' in predictors and 'CHM' in predictors:
        term_names.append('te(SlpPerc,CHM)')
        edfs.append(float(np.sum(edof[idx:idx+k*k])))
        idx += k*k
    if 'SlpPerc' in predictors and NDMI_VAR in predictors:
        term_names.append(f'te(SlpPerc,{NDMI_VAR})')
        edfs.append(float(np.sum(edof[idx:idx+k*k])))
        idx += k*k
    if 'SlpPerc' in predictors and 'DFFM_silo' in predictors:
        term_names.append('te(SlpPerc,DFFM_silo)')
        edfs.append(float(np.sum(edof[idx:idx+k*k])))
        idx += k*k
    return term_names, edfs

def extract_term_statistics(gam, predictors, k):
    """Create a thesis-ready term table with EDF and approximate pyGAM p-values.

    p-values are reported as approximate inferential diagnostics from pyGAM and should
    not be interpreted as causal evidence or used as the sole basis for variable selection.
    """
    term_names, edfs = extract_term_edfs(gam, predictors, k)
    pvals = list(gam.statistics_.get('p_values', []))
    rows = []
    for i, (name, edf) in enumerate(zip(term_names, edfs)):
        p = float(pvals[i]) if i < len(pvals) and np.isfinite(pvals[i]) else np.nan
        rows.append({
            'Term': name,
            'Term_type': 'Tensor interaction' if name.startswith('te(') else 'Smooth',
            'EDF': edf,
            'Approx_p_value': p,
            'Evidence_code': ('***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns') if np.isfinite(p) else ''
        })
    return pd.DataFrame(rows)

def create_5x5_partial_effects_plots(model, data, predictors, landform, output_dir, timestamp):
    """Create 5x5 partial effects plots for all variables"""
    colors = {
        'primary': '#2E86AB',      
        'secondary': '#A23B72',    
        'accent': '#F18F01',       
        'neutral': '#C73E1D',      
        'background': '#F8F9FA',   
        'text': '#2C3E50'          
    }
    fig, axes = plt.subplots(5, 5, figsize=(25, 25))
    axes_flat = axes.flatten()
    for i, pred in enumerate(partial_plot_vars):
        if i >= 25:
            break
        ax = axes_flat[i]
        try:
            if pred not in predictors:
                ax.text(0.5, 0.5, f'{var_names.get(pred, pred)}\nNot in model', ha='center', va='center', transform=ax.transAxes, fontsize=18, color=colors['text'])
                ax.set_title(var_names.get(pred, pred), fontsize=22, fontweight='bold', color=colors['text'], pad=10)
                continue
            pred_idx = predictors.index(pred)
            XX = model.generate_X_grid(term=pred_idx, n=300)
            pdep, confi_95 = model.partial_dependence(term=pred_idx, X=XX, width=0.95)
            _, confi_68 = model.partial_dependence(term=pred_idx, X=XX, width=0.68)
            x_vals = XX[:, pred_idx]
            if pred == 'CHM':
                ax.set_xlim(0, data[pred].max())
            elif pred == 'LAI':
                ax.set_xlim(0, data[pred].max())
            else:
                ax.set_xlim(data[pred].min(), data[pred].max())
            ax.fill_between(x_vals, confi_95[:, 0], confi_95[:, 1], alpha=0.2, color='#BBDEFB', label='95% CI')
            ax.fill_between(x_vals, confi_68[:, 0], confi_68[:, 1], alpha=0.4, color='#64B5F6', label='68% CI')
            ax.plot(x_vals, pdep, linewidth=2.5, color='#263238', label='Fitted effect')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
            title = var_names.get(pred, pred)
            plot_label = chr(97 + i)
            ax.set_title(f'({plot_label}) {title}', fontsize=22, fontweight='bold', color='black', pad=10)
            ax.set_xlabel(title, fontsize=20, color='black')
            ax.set_ylabel('Partial Effect', fontsize=20, color='black')
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=18)
            if pred in ['DwnSlpCurv', 'UpSlpCurv']:
                all_xticks = ax.get_xticks()
                ax.set_xticks(all_xticks)
                ax.set_xticklabels([f'{x:.0f}' if i % 2 == 0 else '' for i, x in enumerate(all_xticks)])
            elif pred in ['CHM', 'LAI', 'DFFM_silo', 'SoilMois_0to10']:
                ax.set_xticks(ax.get_xticks())
                ax.set_yticks(ax.get_yticks())
        except Exception as e:
            print(f"Error plotting {pred}: {e}")
            ax.text(0.5, 0.5, f'Error plotting {pred}', ha='center', va='center', transform=ax.transAxes, fontsize=18, color=colors['text'])
            ax.set_title(var_names.get(pred, pred), fontsize=22, fontweight='bold', color=colors['text'], pad=10)
    for i in range(len(partial_plot_vars), 25):
        axes_flat[i].set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    fig_path = os.path.join(output_dir, f'GAM_partial_effects_5x5_{landform}_{timestamp}.png')
    plt.savefig(fig_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"5x5 Partial effects plots saved: {fig_path}")

def create_thesis_composite_figure(model, data, predictors, landform, output_dir, timestamp):
    """
    Create one thesis-ready 6 x 5 composite figure modelled on the Appendix A layout:
      rows 1-2 : 10 partial-effect plots with 68% and 95% confidence bands
      rows 3-5 : 15 pairwise fitted RdNBR response surfaces
      row 6    : 5 three-dimensional CHM response surfaces, coloured by local sensitivity

    IMPORTANT: the 2-D panels are described as pairwise fitted response surfaces. Only
    CHM x SlpPerc, NDMI x SlpPerc and DFFM x SlpPerc are explicit tensor-product
    terms in the current GAM specification; the other panels visualise the joint fitted
    response of the full model while all remaining predictors are held at their medians.
    """
    from matplotlib import cm, colors as mcolors

    fig = plt.figure(figsize=(17.5, 21.0), constrained_layout=False)
    gs = fig.add_gridspec(6, 5, hspace=0.62, wspace=0.42)

    partial_labels = list('abcdefghij')
    for i, pred in enumerate(partial_plot_vars):
        ax = fig.add_subplot(gs[i // 5, i % 5])
        pred_idx = predictors.index(pred)
        XX = model.generate_X_grid(term=pred_idx, n=300)
        pdep, ci95 = model.partial_dependence(term=pred_idx, X=XX, width=0.95)
        _, ci68 = model.partial_dependence(term=pred_idx, X=XX, width=0.68)
        x = XX[:, pred_idx]

        ax.fill_between(x, ci95[:, 0], ci95[:, 1], color='#BBDEFB', alpha=0.75, linewidth=0)
        ax.fill_between(x, ci68[:, 0], ci68[:, 1], color='#64B5F6', alpha=0.75, linewidth=0)
        ax.plot(x, pdep, color='#263238', linewidth=1.7)
        ax.axhline(0, color='#78909C', linestyle='--', linewidth=0.7, alpha=0.8)
        ax.set_title(f'({partial_labels[i]}) {var_names.get(pred, pred)}', fontsize=10, fontweight='bold')
        ax.set_xlabel(var_names.get(pred, pred), fontsize=8)
        ax.set_ylabel('Partial effect', fontsize=8)
        ax.tick_params(labelsize=7)
        
        vals = pd.to_numeric(data[pred], errors='coerce').dropna().to_numpy()
        if len(vals) > 0:
            q = np.linspace(0.02, 0.98, min(45, len(vals)))
            rug = np.quantile(vals, q)
            ymin, ymax = ax.get_ylim()
            rug_y = ymin + 0.018 * (ymax - ymin)
            ax.plot(rug, np.full_like(rug, rug_y), '|', color='#607D8B', markersize=3.0, alpha=0.45, markeredgewidth=0.55)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def robust_range(series, n=80):
        vals = pd.to_numeric(series, errors='coerce').dropna().to_numpy()
        lo, hi = np.nanpercentile(vals, [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = np.nanmin(vals), np.nanmax(vals)
        return np.linspace(lo, hi, n)

    medians = {v: float(pd.to_numeric(data[v], errors='coerce').median()) for v in predictors}

    pair_surfaces = []
    for var1, var2 in interaction_pairs:
        xr = robust_range(data[var1], 75)
        yr = robust_range(data[var2], 75)
        Xg, Yg = np.meshgrid(xr, yr)
        grid = np.column_stack([np.full(Xg.size, medians[v], dtype=float) for v in predictors])
        grid[:, predictors.index(var1)] = Xg.ravel()
        grid[:, predictors.index(var2)] = Yg.ravel()
        Z = model.predict(grid).reshape(Xg.shape)
        pair_surfaces.append((var1, var2, Xg, Yg, Z))

    all_z = np.concatenate([z.ravel() for *_, z in pair_surfaces])
    vmin, vmax = np.nanpercentile(all_z, [2, 98])
    if vmin == vmax:
        vmin, vmax = np.nanmin(all_z), np.nanmax(all_z)
    levels = np.linspace(vmin, vmax, 18)
    pair_labels = ['k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
    last_cf = None

    for i, (var1, var2, Xg, Yg, Z) in enumerate(pair_surfaces):
        ax = fig.add_subplot(gs[2 + i // 5, i % 5])
        last_cf = ax.contourf(Xg, Yg, Z, levels=levels, cmap='RdYlBu_r', extend='both')
        ax.contour(Xg, Yg, Z, levels=8, linewidths=0.35, alpha=0.45)
        ax.set_title(f'({pair_labels[i]}) {var_names.get(var1,var1)} × {var_names.get(var2,var2)}', fontsize=9, fontweight='bold')
        ax.set_xlabel(var_names.get(var1, var1), fontsize=7.5)
        ax.set_ylabel(var_names.get(var2, var2), fontsize=7.5)
        ax.tick_params(labelsize=6.5)

    if last_cf is not None:
        cax = fig.add_axes([0.925, 0.285, 0.012, 0.315])
        cb = fig.colorbar(last_cf, cax=cax)
        cb.set_label('Fitted RdNBR', fontsize=8)
        cb.ax.tick_params(labelsize=7)

    labels3d = ['aa','ab','ac','ad','ae']
    sensitivity_mappable = cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap='viridis')
    sensitivity_mappable.set_array([])

    for i, (var1, var2) in enumerate(ch_3d_interactions):
        ax = fig.add_subplot(gs[5, i], projection='3d')
        xr = robust_range(data[var1], 55)
        yr = robust_range(data[var2], 55)
        Xg, Yg = np.meshgrid(xr, yr)
        grid = np.column_stack([np.full(Xg.size, medians[v], dtype=float) for v in predictors])
        grid[:, predictors.index(var1)] = Xg.ravel()
        grid[:, predictors.index(var2)] = Yg.ravel()
        Z = model.predict(grid).reshape(Xg.shape)

        dy = float(yr[1] - yr[0]) if len(yr) > 1 else 1.0
        dx = float(xr[1] - xr[0]) if len(xr) > 1 else 1.0
        gz_y, gz_x = np.gradient(Z, dy, dx)
        sens = np.sqrt(gz_x**2 + gz_y**2)
        smin, smax = np.nanmin(sens), np.nanmax(sens)
        sens01 = (sens - smin) / (smax - smin) if smax > smin else np.zeros_like(sens)

        ax.plot_surface(Xg, Yg, Z, facecolors=cm.viridis(sens01), linewidth=0, antialiased=True, shade=False, alpha=0.95)
        ax.set_title(f'({labels3d[i]}) {var_names.get(var1,var1)} × {var_names.get(var2,var2)}', fontsize=8.5, fontweight='bold', pad=2)
        ax.set_xlabel(var_names.get(var1,var1), fontsize=6.5, labelpad=2)
        ax.set_ylabel(var_names.get(var2,var2), fontsize=6.5, labelpad=2)
        ax.set_zlabel('Fitted RdNBR', fontsize=6.5, labelpad=2)
        ax.tick_params(labelsize=5.5, pad=0)
        ax.view_init(elev=28, azim=-55)

    cax2 = fig.add_axes([0.40, 0.035, 0.22, 0.008])
    cb2 = fig.colorbar(sensitivity_mappable, cax=cax2, orientation='horizontal')
    cb2.set_ticks([0, 1])
    cb2.set_ticklabels(['Low', 'High'])
    cb2.set_label('Local sensitivity', fontsize=7)
    cb2.ax.tick_params(labelsize=6)

    fig.suptitle(f'GAM fitted relationships and pairwise response surfaces — {landform}', fontsize=14, fontweight='bold', y=0.995)
    fig.subplots_adjust(left=0.055, right=0.91, top=0.975, bottom=0.065)

    png_path = os.path.join(output_dir, f'GAM_thesis_composite_{landform}_{timestamp}.png')
    pdf_path = os.path.join(output_dir, f'GAM_thesis_composite_{landform}_{timestamp}.pdf')
    fig.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Thesis composite GAM figure saved: {png_path}')
    print(f'Thesis composite GAM figure saved: {pdf_path}')

print(f"\n{'='*60}")
print(f"Running GAM V10 for OVERALL DATA (all landforms combined)")
print(f"{'='*60}")

df_overall = data.dropna(subset=[response_var] + all_vars)

if len(df_overall) >= 50:
    print(f"Overall data points: {len(df_overall)}")
    model_overall = fit_gam_model(df_overall, response_var, all_vars)
    if model_overall is not None:
        X = df_overall[all_vars].values
        y = df_overall[response_var].values
        y_pred = model_overall.predict(X)
        residuals = y - y_pred
        rmse, mae, r2 = calculate_performance_metrics(y, y_pred)
        try:
            total_deviance = np.sum((y - np.mean(y))**2)
            residual_deviance = np.sum(residuals**2)
            dev_expl = (1 - residual_deviance / total_deviance) * 100
            term_names, edfs = extract_term_edfs(model_overall, all_vars, 20)
            metrics_df = pd.DataFrame({
                'Metric': ['RMSE', 'MAE', 'R²', 'Deviance Explained (%)', 'Sample Size'],
                'Value': [rmse, mae, r2, dev_expl, len(df_overall)]
            })
            metrics_path = os.path.join(output_dir, f'GAM_model_metrics_Overall_{timestamp}.csv')
            metrics_df.to_csv(metrics_path, index=False)
            print(f"Model metrics saved: {metrics_path}")
            edf_df = pd.DataFrame({
                'Term': term_names,
                'EDF': edfs
            })
            edf_path = os.path.join(output_dir, f'GAM_smooth_summary_Overall_{timestamp}.csv')
            edf_df.to_csv(edf_path, index=False)
            print(f"Smooth term summary saved: {edf_path}")
            term_stats_df = extract_term_statistics(model_overall, all_vars, 20)
            term_stats_path = os.path.join(output_dir, f'GAM_term_statistics_Overall_{timestamp}.csv')
            term_stats_df.to_csv(term_stats_path, index=False)
            print(f"GAM term statistics saved: {term_stats_path}")
            print(f"\nModel Performance (Overall):")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R²: {r2:.4f}")
            print(f"Deviance Explained: {dev_expl:.2f}%")
            print(f"Sample Size: {len(df_overall)}")
        except Exception as e:
            print(f"Error calculating model statistics: {e}")
        create_diagnostic_plots(model_overall, y, y_pred, residuals, 'Overall', output_dir, timestamp)
        create_5x5_partial_effects_plots(model_overall, df_overall, all_vars, 'Overall', output_dir, timestamp)
        create_2d_interaction_plots(model_overall, df_overall, all_vars, 'Overall', output_dir, timestamp)
        create_3d_ch_interaction_plots(model_overall, df_overall, all_vars, 'Overall', output_dir, timestamp)
        create_thesis_composite_figure(model_overall, df_overall, all_vars, 'Overall', output_dir, timestamp)
    else:
        print("Failed to fit GAM model for overall data")
else:
    print(f"Insufficient data points ({len(df_overall)}) for overall analysis")

print(f"\n{'='*60}")
print(f"Running GAM V10 for PER-LANDFORM ANALYSIS")
print(f"{'='*60}")

landforms = data['LF4Cls'].unique()
print(f"Landforms found: {landforms}")

for lf in landforms:
    print(f"\n{'='*40}")
    print(f"Analyzing landform: {lf}")
    print(f"{'='*40}")
    df = data[data['LF4Cls'] == lf].dropna(subset=[response_var] + all_vars)
    if len(df) >= 50:
        print(f"Data points for {lf}: {len(df)}")
        model = fit_gam_model(df, response_var, all_vars)
        if model is not None:
            X = df[all_vars].values
            y = df[response_var].values
            y_pred = model.predict(X)
            residuals = y - y_pred
            rmse, mae, r2 = calculate_performance_metrics(y, y_pred)
            try:
                total_deviance = np.sum((y - np.mean(y))**2)
                residual_deviance = np.sum(residuals**2)
                dev_expl = (1 - residual_deviance / total_deviance) * 100
                term_names, edfs = extract_term_edfs(model, all_vars, 20)
                metrics_df = pd.DataFrame({
                    'Metric': ['RMSE', 'MAE', 'R²', 'Deviance Explained (%)', 'Sample Size'],
                    'Value': [rmse, mae, r2, dev_expl, len(df)]
                })
                metrics_path = os.path.join(output_dir, f'GAM_model_metrics_{lf}_{timestamp}.csv')
                metrics_df.to_csv(metrics_path, index=False)
                print(f"Model metrics saved: {metrics_path}")
                edf_df = pd.DataFrame({
                    'Term': term_names,
                    'EDF': edfs
                })
                edf_path = os.path.join(output_dir, f'GAM_smooth_summary_{lf}_{timestamp}.csv')
                edf_df.to_csv(edf_path, index=False)
                print(f"Smooth term summary saved: {edf_path}")
                term_stats_df = extract_term_statistics(model, all_vars, 20)
                term_stats_path = os.path.join(output_dir, f'GAM_term_statistics_{lf}_{timestamp}.csv')
                term_stats_df.to_csv(term_stats_path, index=False)
                print(f"GAM term statistics saved: {term_stats_path}")
                print(f"\nModel Performance ({lf}):")
                print(f"RMSE: {rmse:.4f}")
                print(f"MAE: {mae:.4f}")
                print(f"R²: {r2:.4f}")
                print(f"Deviance Explained: {dev_expl:.2f}%")
                print(f"Sample Size: {len(df)}")
            except Exception as e:
                print(f"Error calculating model statistics: {e}")
            create_diagnostic_plots(model, y, y_pred, residuals, lf, output_dir, timestamp)
            create_5x5_partial_effects_plots(model, df, all_vars, lf, output_dir, timestamp)
            create_2d_interaction_plots(model, df, all_vars, lf, output_dir, timestamp)
            create_3d_ch_interaction_plots(model, df, all_vars, lf, output_dir, timestamp)
        else:
            print(f"Failed to fit GAM model for {lf}")
    else:
        print(f"Insufficient data points ({len(df)}) for {lf} analysis")

end_time = time.time()
total_time = end_time - start_time
print(f"\n{'='*60}")
print(f"GAM V10 Analysis Complete!")
print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"Output files saved in: {output_dir}")
summary_path = OUTPUT_DIR / f'GAM_run_summary_{timestamp}.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('Fire severity GAM analysis\n')
    f.write(f'Data file: {file_path}\n')
    f.write(f'Response: {response_var} (continuous RdNBR)\n')
    f.write(f'NDMI field: {NDMI_VAR}\n')
    f.write('Predictors: ' + ', '.join(all_vars) + '\n')
    f.write('Explicit tensor interactions: CHM x SlpPerc; NDMI x SlpPerc; DFFM x SlpPerc\n')
    f.write('Partial-effect uncertainty: 68% and 95% intervals; rug marks show predictor data support.\n')
    f.write('Pairwise response surfaces hold all other predictors at their medians and use 1st-99th percentile plotting ranges.\n')
    f.write('Term p-values are approximate pyGAM diagnostics and should not be interpreted causally.\n')
    f.write(f'Execution time (minutes): {total_time/60:.2f}\n')
print(f"Run summary saved: {summary_path}")
print(f"{'='*60}")
