# -*- coding: utf-8 -*-
'''
Author: Xiaoyu Zhang (xiaoyu.zhang@aces.su.se)
Date: 2026-05-27 16:39:03
LastEditTime: 2026-06-03 16:01:49
Description: Functions for SPLINTER:
             - generate valid knot configurations
             - change point detection
             - GAM modeling and analysis

'''


import os
from datetime import datetime
import warnings
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats as sp_stats
from itertools import combinations, product
import ruptures as rpt
import time
import gc



#  0. Global Helper Functions

def generate_valid_knot_configurations(t, time_span, breaks_df, StartPad, EndPad, MaxKnotPeriod, MinKnotInterval):
    """
    Generate valid knot configurations using possible knot positions from breaks_df:
    - All combinations of possible knot positions from breaks_df that fulfill requirements

    Parameters:
    -----------
    t : array
        Time values
    time_span : float
        Total time span
    breaks_df : pd.DataFrame
        DataFrame containing possible breakpoint positions
    Returns:
    --------
    list of arrays : Valid knot configurations
    """
    from itertools import combinations

    # Get possible knot positions from breaks_df
    try:
        if breaks_df is not None and not breaks_df.empty:
            possible_knots = sorted(breaks_df['Decimal Year'].astype(float).values)
        else:
            return [[]]
    except:
        return [[]]
    # Exclude knots within StartPad years of start and EndPad years of end
        start_limit = t[0] + StartPad
        end_limit = t[-1] - EndPad
        possible_knots = [k for k in possible_knots if k >= start_limit and k <= end_limit]

    max_knots = int(time_span / MaxKnotPeriod)
    min_spacing = MinKnotInterval
    valid_configs = [[]]  # Always include empty configuration (no interior knots)

    # For each number of knots from 1 to max_knots
    for n_knots in range(1, max_knots + 1):
        for knot_combo in combinations(possible_knots, n_knots):
            # Check minimum spacing between knots (endpoints excluded)
            if all((knot_combo[j+1] - knot_combo[j]) >= min_spacing for j in range(len(knot_combo)-1)):
                valid_configs.append(np.array(knot_combo))
    return valid_configs



#  1. Function for change point detection

def detect_changepoints(data_df, data_label, use_data_driven=True, input_subst=None, station_label=None, 
                       visualization=True):
    """
    Perform change point detection on time series data using PELT algorithm with RBF kernel.
    
    Supports two modes:
    1. Data-driven (default): Automatically calculates penalty, min_size, jump based on data size
    2. Fixed parameters: Use defalut penalty value, min_size, jump
    
    Parameters:
    -----------
    data_df : pd.DataFrame
        Input dataframe containing 'ln_c' (log concentration) and 'time_y' (decimal year) columns
    data_label : str
        Label for the data type, e.g., 'raw' or 'cleaned' (used in titles and output names)
    input_subst : str, optional
        Component name for plot title
    station_label : str, optional
        Station label for plot title
    use_data_driven : bool, optional
        If True (default), automatically calculate pen, min_size, jump based on log2(n)
        If False, use provided pen_value, min_size, jump parameters
    visualization : bool, optional
        Whether to plot the detected breakpoints (default=True)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'breaks_df': DataFrame with detected breakpoints (Decimal Year)
        - 'break_years_data': List of breakpoint years
        - 'lnc_data': Array of log concentrations
        - 'x_data': Array of time values (decimal years)
        - 'bkps_raw': Raw breakpoint indices from PELT algorithm
        - 'pelt_params': Dict with actual PELT parameters used
    """
    from sklearn.cluster import KMeans
    
    # Extract data
    lnc_data = data_df['ln_c'].dropna().values
    x_data = data_df.loc[data_df['ln_c'].notna(), 'time_y'].values
    
    n = len(lnc_data)
    
    # Determine PELT parameters
    if use_data_driven:
        # Data-driven approach: auto-calculate parameters based on log2(n)
        log2n = np.log2(n)
        min_size = max(2, int(0.5 * log2n))
        pen_value = 0.5 * log2n
        jump_size = max(1, int(0.5 * log2n))
        
        print(f"\n Data-Driven PELT Parameters:")
        print(f"   n={n}, log₂(n)={log2n:.2f}")
        print(f"   min_size={min_size}, pen={pen_value:.2f}, jump={jump_size}")
    else:
        # Fixed parameters: use provided values or defaults
        pen_value = 3
        min_size = 2
        jump_size = 1
        
        print(f"\n Fixed PELT Parameters:")
        print(f"   n={n}")
        print(f"   min_size={min_size}, pen={pen_value}, jump={jump_size}")
    
    # Stack for PELT (requires 2D input)
    signal = np.column_stack((lnc_data, x_data))
    
    # Run PELT algorithm with RBF kernel
    algo = rpt.Pelt(model="rbf", min_size=min_size, jump=jump_size).fit(signal)
    bkps_raw = algo.predict(pen=pen_value)
    
    # Convert breakpoint indices to decimal years
    break_years_data = [float(x_data[idx-1]) for idx in bkps_raw[:-1]]
    
    # Create DataFrame with results
    breaks_df = pd.DataFrame({
        'Breakpoint #': range(1, len(break_years_data) + 1),
        'Decimal Year': [float(by) for by in break_years_data]
    })
    
    # Plot if requested
    if visualization:
        fig = plt.figure(figsize=(12, 4))
        plt.plot(x_data, lnc_data, label=f'ln_c ({data_label})')
        
        for i, by in enumerate(break_years_data):
            plt.axvline(x=by, color='red', linestyle='--', alpha=0.7,
                       label='Breakpoint' if i == 0 else None)
        
        title_str = f'{input_subst} – Change Point Detection on ln_c ({data_label})'
        if station_label:
            title_str += f'\n{station_label}'
        plt.title(title_str)
        plt.xlabel('Decimal Year (time_y)')
        plt.ylabel('ln_c')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close(fig)
    
    # Print results
    breaks_text = breaks_df.to_string(index=False)
    print(f'✓ Detected {len(break_years_data)} breakpoints (by decimal year) [{data_label}]:')
    print(breaks_text)
    print("")
    
    # Reduce to 20 breakpoints if too many using KMeans
    if len(breaks_df) > 20:
        breakpoints = breaks_df["Decimal Year"].astype(float).values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=20, random_state=0).fit(breakpoints)
        refined_breaks = sorted([float(center[0]) for center in kmeans.cluster_centers_])
        
        breaks_df = pd.DataFrame({'Decimal Year': refined_breaks})
        print(f'Too many breakpoints detected. Using KMeans clustering to reduce to 20 breakpoints [{data_label}]:')
        print(breaks_df.to_string(index=False))
    
    return {
        'breaks_df': breaks_df,
        'break_years_data': break_years_data,
        'lnc_data': lnc_data,
        'x_data': x_data,
        'bkps_raw': bkps_raw,
        'pelt_params': {
            'n': n,
            'min_size': min_size,
            'pen': pen_value,
            'jump': jump_size,
            'use_data_driven': use_data_driven
        }
    }
    


#  2. Functions for GAM modeling and analysis

def fourier_basis_1year(t, n_harmonics):
    X_list, labels = [], []
    omega = 2 * np.pi / 1.0
    for k in range(1, n_harmonics + 1):
        X_list.append(np.sin(k * omega * t))
        labels.append(f'sin_1y_h{k}')
        X_list.append(np.cos(k * omega * t))
        labels.append(f'cos_1y_h{k}')
    return np.column_stack(X_list), labels

def create_spline_basis(t, knot_positions):
    if len(knot_positions) == 0:
        return np.column_stack([np.ones_like(t), t])
    basis = [np.ones_like(t), t]
    for knot in knot_positions:
        basis.append(np.maximum(t - knot, 0))
    return np.column_stack(basis)

def fit_gam_model(t, y, n_harmonics, knot_positions):
    X_components = []
    component_indices = {}
    idx = 0
    if n_harmonics > 0:
        X_seas, _ = fourier_basis_1year(t, n_harmonics)
        X_components.append(X_seas)
        component_indices['seasonal'] = (idx, idx + X_seas.shape[1])
        idx += X_seas.shape[1]
    else:
        component_indices['seasonal'] = None
    X_spline = create_spline_basis(t, knot_positions)
    X_components.append(X_spline)
    component_indices['trend'] = (idx, idx + X_spline.shape[1])
    X = np.column_stack(X_components)
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    y_pred = X @ coeffs
    residuals = y - y_pred
    n, p = len(y), X.shape[1]
    rss = np.sum(residuals**2)
    bic = n * np.log(rss / n) + p * np.log(n)
    aic = n * np.log(rss / n) + 2 * p
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - rss / ss_tot if ss_tot != 0 else np.nan
    rmse = np.sqrt(rss / n)
    components = {}
    for name, (s, e) in component_indices.items():
        if name == 'seasonal' and s is None:
            components[name] = np.zeros_like(t)
        else:
            Xc = X[:, s:e]
            components[name] = Xc @ coeffs[s:e]
    return {
        'coeffs': coeffs, 'y_pred': y_pred, 'residuals': residuals,
        'bic': bic, 'aic': aic, 'r2': r2, 'rmse': rmse,
        'n_params': p, 'components': components, 'knot_positions': knot_positions,
        'component_indices': component_indices, 'X': X,
        'n_harmonics': n_harmonics, 'n_knots': len(knot_positions)
    }

def run_gam_model(data_df, 
                  breaks_df,
                  model_name,             # str: 'Model1' or 'Model2'
                  data_description,       # str: 'Raw Data' or 'Cleaned Data'
                  input_subst,
                  station_label,
                  station_label_fn,
                  MaxKnotPeriod,
                  MinKnotInterval,
                  StartPad,
                  EndPad,
                  LastKnotYear=None,
                  _user_EndPad=None,
                  harmonics_range=None,
                  save_plots=False):
    """
    execute GAM modeling, print analysis results and plots.
    return: updated dataframe (with added residuals and other columns) and best model dictionary.
    """
    if harmonics_range is None:
        harmonics_range = [0, 1, 2]

    print("=" * 80)
    print(f"GAM DECOMPOSITION WITH ITERATIVE MODEL SELECTION ({model_name}: {data_description})")
    print("=" * 80)

    # prepare data
    gam_data = data_df.copy().sort_values('time_y').reset_index(drop=True)
    t = gam_data['time_y'].values
    y = gam_data['ln_c'].values
    time_span = t.max() - t.min()
    n_obs = len(t)

    # process LastKnotYear / EndPad
    if LastKnotYear is not None:
        if LastKnotYear < t[-1]:
            EndPad = t[-1] - LastKnotYear
            print(f"LastKnotYear = {LastKnotYear} → EndPad = {t[-1]:.4f} − {LastKnotYear} = {EndPad:.4f} years ({model_name})")
        else:
            EndPad = _user_EndPad if _user_EndPad is not None else EndPad
            print(f"LastKnotYear = {LastKnotYear} is at or beyond data end ({t[-1]:.4f}); reverting to user EndPad = {EndPad:.4f} yr ({model_name})")

    print(f"Time span: {time_span:.2f} years")
    print(f"Number of observations: {n_obs}")
    print(f"Time range: {t.min():.2f} to {t.max():.2f}")

    # generate valid knot configurations
    valid_knot_configs = generate_valid_knot_configurations(t, time_span, breaks_df, StartPad, EndPad, MaxKnotPeriod, MinKnotInterval)

    print(f"\nTesting {len(harmonics_range)} seasonal harmonic options")
    print(f"Testing {len(valid_knot_configs)} valid knot configurations")
    total_models = len(harmonics_range) * len(valid_knot_configs)
    print(f"Total models to evaluate: {total_models}")

    start_time = time.time()
    all_models = []
    for n_harm in harmonics_range:
        for knot_config in valid_knot_configs:
            try:
                model = fit_gam_model(t, y, n_harm, knot_config)
                light = {k: model[k] for k in ['bic', 'aic', 'r2', 'rmse', 'n_params', 'n_harmonics', 'n_knots', 'knot_positions']}
                light.update({'n_harmonics': n_harm, 'n_knots': len(knot_config)})
                all_models.append(light)
            except Exception as e:
                print(f"  Model failed (h={n_harm}, k={len(knot_config)}): {str(e)}")
                continue
    end_time = time.time()
    print(f"\n✓ Fitted {len(all_models)} models successfully in {end_time - start_time:.3f} seconds")

    all_models_sorted = sorted(all_models, key=lambda x: x['bic'])
    print("\n" + "=" * 80)
    print(f"TOP 10 MODELS BY BIC ({model_name})")
    print("=" * 80)
    print(f"{'Rank':<6} {'Harmonics':<12} {'Knots':<8} {'BIC':<12} {'R²':<10} {'RMSE':<10}")
    print("-" * 80)
    for i, m in enumerate(all_models_sorted[:10]):
        print(f"{i+1:<6} {m['n_harmonics']:<12} {m['n_knots']:<8} {m['bic']:<12.2f} {m['r2']:<10.6f} {m['rmse']:<10.6f}")

    best_light = all_models_sorted[0]
    best_model = fit_gam_model(t, y, best_light['n_harmonics'], best_light['knot_positions'])
    best_model['n_harmonics'] = best_light['n_harmonics']
    best_model['n_knots'] = best_light['n_knots']

    print("\n" + "=" * 80)
    print(f"BEST MODEL SELECTED ({model_name})")
    print("=" * 80)
    print(f"Seasonal harmonics (1-year period): {best_model['n_harmonics']}")
    print(f"Spline knots: {best_model['n_knots']}")
    print(f"Total parameters: {best_model['n_params']}")
    print(f"BIC: {best_model['bic']:.2f}")
    print(f"AIC: {best_model['aic']:.2f}")
    print(f"R²: {best_model['r2']:.6f}")
    print(f"RMSE: {best_model['rmse']:.6f}")

    # ---------- save fitted values to gam_data ----------
    gam_data['gam_fitted'] = best_model['y_pred']
    gam_data['gam_residuals'] = best_model['residuals']
    gam_data['gam_seasonal'] = best_model['components']['seasonal']
    gam_data['gam_trend'] = best_model['components']['trend']

    # ---------- confidence intervals ----------
    print("\n" + "=" * 80)
    print(f"CALCULATING 95% CONFIDENCE INTERVALS ({model_name})")
    print("=" * 80)
    n = len(y)
    p = best_model['n_params']
    sigma2 = np.sum(best_model['residuals']**2) / (n - p)
    X = best_model['X']
    cov_matrix = sigma2 * np.linalg.inv(X.T @ X)
    best_model['cov_matrix'] = cov_matrix
    se_fitted = np.sqrt(np.diag(X @ cov_matrix @ X.T))
    t_crit = sp_stats.t.ppf(0.975, n - p)
    best_model['t_critical'] = t_crit
    best_model['ci_lower'] = best_model['y_pred'] - t_crit * se_fitted
    best_model['ci_upper'] = best_model['y_pred'] + t_crit * se_fitted
    best_model['se_fitted'] = se_fitted

    # trend component confidence intervals
    trend_start, trend_end = best_model['component_indices']['trend']
    X_trend = X[:, trend_start:trend_end]
    cov_trend = cov_matrix[trend_start:trend_end, trend_start:trend_end]
    se_trend = np.sqrt(np.diag(X_trend @ cov_trend @ X_trend.T))
    best_model['trend_ci_lower'] = best_model['components']['trend'] - t_crit * se_trend
    best_model['trend_ci_upper'] = best_model['components']['trend'] + t_crit * se_trend
    print(f"✓ 95% confidence intervals calculated for {model_name}")
    print(f"  Critical t-value (df={n-p}): {t_crit:.4f}")

    # ---------- detailed component analysis ----------
    print("\n" + "=" * 80)
    print(f"DETAILED COMPONENT ANALYSIS ({model_name})")
    print("=" * 80)
    trend_comp = best_model['components']['trend']
    deriv = np.gradient(trend_comp, t)
    gam_data['gam_trend_derivative'] = deriv

    # coefficients significance
    coeffs = best_model['coeffs']
    std_errors = np.sqrt(np.diag(cov_matrix))
    t_stats = coeffs / std_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p))
    sig = p_values < 0.05

    print("\n" + "-" * 80)
    print(f"COMPONENT 1: SEASONAL (1-YEAR PERIOD) ({model_name})")
    print("-" * 80)
    if best_model['component_indices']['seasonal'] is not None:
        s, e = best_model['component_indices']['seasonal']
        seas_coeffs = coeffs[s:e]
        seas_pvals = p_values[s:e]
        seas_sig = sig[s:e]
        n_h = best_model['n_harmonics']
        print(f"Number of harmonics: {n_h}")
        print(f"Total parameters: {len(seas_coeffs)}")
        print(f"Significant parameters (p < 0.05): {np.sum(seas_sig)}")
        idx = 0
        for k in range(1, n_h + 1):
            sin_c, cos_c = seas_coeffs[idx], seas_coeffs[idx+1]
            amp = np.sqrt(sin_c**2 + cos_c**2)
            phase = np.arctan2(sin_c, cos_c)
            print(f"  Harmonic {k}: sin={sin_c:+.6f}, cos={cos_c:+.6f}, amplitude={amp:.6f}, phase={phase:.4f} rad")
            idx += 2
    else:
        print("No seasonal component in model")

    print("\n" + "-" * 80)
    print(f"COMPONENT 2: TREND (LINEAR SPLINE) ({model_name})")
    print("-" * 80)
    ts, te = best_model['component_indices']['trend']
    trend_coeffs = coeffs[ts:te]
    trend_pvals = p_values[ts:te]
    trend_sig = sig[ts:te]
    print(f"Total parameters: {len(trend_coeffs)}")
    print(f"Significant parameters (p < 0.05): {np.sum(trend_sig)}")
    print(f"Number of knots: {best_model['n_knots']}")
    print(f"  Intercept: {trend_coeffs[0]:+.6f} (p={trend_pvals[0]:.4f}){'*' if trend_sig[0] else ''}")
    print(f"  Linear term: {trend_coeffs[1]:+.6f} (p={trend_pvals[1]:.4f}){'*' if trend_sig[1] else ''}")
    for i, knot in enumerate(best_model['knot_positions']):
        print(f"  Knot at {knot:.2f}: coeff={trend_coeffs[2+i]:+.6f} (p={trend_pvals[2+i]:.4f}){'*' if trend_sig[2+i] else ''}")
    print(f"\nTrend derivative: mean={np.mean(deriv):.6f}, median={np.median(deriv):.6f}, range=[{np.min(deriv):.6f}, {np.max(deriv):.6f}] ln/year")

    # residual diagnostics
    res = best_model['residuals']
    if len(res) >= 50:
        stat, p_norm = stats.kstest(res, 'norm', args=(res.mean(), res.std()))
        test_name = "Kolmogorov-Smirnov"
        stat_label = "D"
    else:
        stat, p_norm = stats.shapiro(res)
        test_name = "Shapiro-Wilk"
        stat_label = "W"
    print("\n" + "-" * 80)
    print(f"MODEL DIAGNOSTICS ({model_name})")
    print("-" * 80)
    print(f"Total parameters: {p}")
    print(f"Significant parameters (p < 0.05): {np.sum(sig)}")
    print(f"Residual Normality ({test_name}): {stat_label}={stat:.4f}, p={p_norm:.4f} → {'Normal' if p_norm > 0.05 else 'Not normal'}")

    # ---------- visualization ----------
    fig1, axes = plt.subplots(5, 1, figsize=(16, 15))
    fig1.suptitle(f'{input_subst} – GAM Decomposition - {model_name} - {data_description} (h={best_model["n_harmonics"]}, k={best_model["n_knots"]})\n{station_label}',
                  fontsize=16, fontweight='bold')
    axes[0].scatter(t, y, alpha=0.4, s=20, c='black', label='Observed')
    axes[0].plot(t, best_model['y_pred'], 'r-', lw=2, label='GAM Fit')
    axes[0].set_ylabel('ln(Concentration)')
    axes[0].set_title(f'{data_description} vs GAM Fit (R²={best_model["r2"]:.4f})')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(t, best_model['components']['seasonal'], 'b-', lw=1.5)
    axes[1].axhline(0, color='gray', ls='--', alpha=0.5)
    axes[1].set_ylabel('Seasonal'); axes[1].set_title(f'Seasonal Component (1-year Fourier, h={best_model["n_harmonics"]})'); axes[1].grid(alpha=0.3)
    axes[2].plot(t, best_model['components']['trend'], 'r-', lw=2)
    for knot in best_model['knot_positions']:
        axes[2].axvline(knot, color='orange', ls=':', alpha=0.7)
    axes[2].set_ylabel('Trend'); axes[2].set_title(f'Trend Component (Linear Spline, {best_model["n_knots"]} knots)'); axes[2].grid(alpha=0.3)
    axes[3].plot(t, deriv, 'purple', lw=1.5)
    axes[3].axhline(0, color='red', ls='--', lw=2)
    axes[3].set_ylabel('Derivative (ln/year)'); axes[3].set_title('Trend Derivative'); axes[3].grid(alpha=0.3)
    axes[4].scatter(t, best_model['residuals'], alpha=0.5, s=20, c='purple')
    axes[4].axhline(0, color='red', ls='--', lw=2)
    axes[4].set_ylabel('Residuals'); axes[4].set_xlabel('Time (years)'); axes[4].set_title(f'Residuals (RMSE={best_model["rmse"]:.6f})'); axes[4].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    if save_plots:
        from splinter_functions import save_plot
        save_plot(fig1, f'GAM_{model_name}_Decomposition_{input_subst}_{station_label_fn}')
    plt.close(fig1)

    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 10))
    fig2.suptitle(f'{input_subst} – GAM Diagnostic Plots - {model_name} - {data_description}\n{station_label}',
                  fontsize=16, fontweight='bold')
    axes2[0,0].scatter(y, best_model['y_pred'], alpha=0.5, s=20, c='blue')
    axes2[0,0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes2[0,0].set_xlabel('Observed ln(Concentration)'); axes2[0,0].set_ylabel('Fitted Values')
    axes2[0,0].set_title(f'R²={best_model["r2"]:.4f}, RMSE={best_model["rmse"]:.6f}'); axes2[0,0].grid(alpha=0.3)
    axes2[0,1].scatter(t, best_model['residuals'], alpha=0.5, s=20, c='blue')
    axes2[0,1].axhline(0, color='red', ls='--', lw=2)
    axes2[0,1].set_xlabel('Time (years)'); axes2[0,1].set_ylabel('Residuals')
    axes2[0,1].set_title(f'Residuals over Time (std={np.std(best_model["residuals"]):.6f})'); axes2[0,1].grid(alpha=0.3)
    axes2[1,0].hist(best_model['residuals'], bins=40, alpha=0.7, color='blue', edgecolor='black')
    axes2[1,0].set_xlabel('Residual Value'); axes2[1,0].set_ylabel('Frequency')
    axes2[1,0].set_title(f'Residual Distribution\n{test_name} p={p_norm:.4f}'); axes2[1,0].axvline(0, color='red', ls='--', lw=2); axes2[1,0].grid(alpha=0.3, axis='y')
    stats.probplot(best_model['residuals'], dist="norm", plot=axes2[1,1])
    axes2[1,1].set_title('Q-Q Plot of Residuals'); axes2[1,1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    if save_plots:
        save_plot(fig2, f'GAM_{model_name}_Diagnostics_{input_subst}_{station_label_fn}')
    plt.close(fig2)

    # ---------- save residuals to original data frame ----------
    # choose a column name that won't conflict with existing columns
    col_resid = f'{model_name}_residuals'
    data_df[col_resid] = np.nan
    data_df.loc[gam_data.index, col_resid] = best_model['residuals']
    print("\n" + "=" * 80)
    print(f"RESIDUALS SAVED TO DATA DATAFRAME ({model_name})")
    print("=" * 80)
    print(f"Column '{col_resid}' added to data dataframe")
    print(f"Non-NaN values: {data_df[col_resid].notna().sum()}")
    print("=" * 80)
    
    # ---------- summary ----------
    print("\n" + "=" * 80)
    print(f"SUMMARY ({model_name})")
    print("=" * 80)
    print(f"✓ Model components:")
    print(f"  - Seasonal: 1-year Fourier with {best_model['n_harmonics']} harmonics")
    print(f"  - Trend: Linear spline with {best_model['n_knots']} knots")
    print(f"✓ Model performance:")
    print(f"  - BIC: {best_model['bic']:.2f}")
    print(f"  - R²: {best_model['r2']:.6f}")
    print(f"  - RMSE: {best_model['rmse']:.6f}")
    print(f"✓ Execution time: {end_time - start_time:.3f} seconds")
    print("=" * 80)

    # clean up large objects to save memory
    best_model.pop('X', None)
    gc.collect()

    return data_df, best_model

