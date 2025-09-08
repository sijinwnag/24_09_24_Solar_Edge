"""
Multi-Orientation Solar Array Mismatch Analysis
Enhanced version of the original generator with per-timestamp K-means clustering

This module implements multi-orientation solar array analysis by grouping modules
using K-means clustering based on panel current values at each timestamp, then 
calculating series power within each group and summing across groups for improved 
mismatch loss estimation.

Author: PV Engineer & Software Engineer Agents
Date: September 2025
"""

import os
import sys
import time
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pvlib
import imageio
import datetime
from datetime import timedelta
import scipy.constants as const
import matplotlib.dates as mdates
from typing import Dict, List, Tuple, Optional, Union
import warnings
from sklearn.cluster import KMeans

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Multi-orientation site configuration (from analysis of existing data)
MULTI_ORIENTATION_SITES = {
    '3455043': 3,
    '4111492': 4,
    '4111800': 4,
    '4118327': 4,
    '3794347': 6,
    '4173851': 4
}

# Directory configuration
DATA_DIR = r"C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\24_09_24_Solar_Edge\Data"
BASE_DIR = r"C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\24_09_24_Solar_Edge\Results\25_06_24_Results"
SUMMARY_DIR = r"C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\24_09_24_Solar_Edge\Data\25_05_01_Newsites_summary.xlsx"

# Plot configuration (matching existing style)
Y_LIMIT_MODULE = (0, 15)
X_LIMIT_MODULE = (0, 60)
Y_LIMIT_INVERTER = (0, 17)
X_LIMIT_INVERTER = (0, 1200)

# Plotting style parameters
AXIS_LABEL_SIZE = 20
AXIS_NUM_SIZE = 20
TEXT_SIZE = 20
TITLE_SIZE = 22
FIGURE_SIZE = (6, 6)
LONG_HOZ_FIGSIZE = (12, 6)  # Match original single-orientation dimensions

# Physics parameters
USE_DYNAMIC_VTH = True
USE_A_T = True  # Use ambient temperature instead of panel temperature
BOLTZMANN_CONSTANT = const.Boltzmann
ELECTRON_CHARGE = const.e

# Timestamp formats for parsing
TIMESTAMP_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M",
    "%Y-%d-%m %H:%M:%S",
    None  # Let pandas infer
]

# Seasonal mapping for hemispheres
SEASON_MONTHS_SOUTH = {
    'summer': ['december', 'january', 'february'],
    'autumn': ['march', 'april', 'may'],
    'winter': ['june', 'july', 'august'],
    'spring': ['september', 'october', 'november']
}

SEASON_MONTHS_NORTH = {
    'summer': ['june', 'july', 'august'],
    'autumn': ['september', 'october', 'november'],
    'winter': ['december', 'january', 'february'],
    'spring': ['march', 'april', 'may']
}

# ============================================================================
# UTILITY FUNCTIONS (FROM ORIGINAL WORKFLOW)
# ============================================================================

def I0(I: float, V: float, Rs: float, Rsh: float, n: float, N: int, vth: float) -> float:
    """Calculate dark saturation current from MPP data."""
    exp_term = np.exp(-(V + I * Rs) / (n * N * vth))
    frac_term = n * N * vth / V
    numerator = I * (1 + Rs / Rsh) - V / Rsh
    denominator = 1 - I * Rs / V
    return numerator / denominator * frac_term * exp_term


def IL(I: float, V: float, Rs: float, Rsh: float, n: float, N: int, vth: float, I0_val: float) -> float:
    """Calculate light-generated current from MPP data and I0."""
    first_term = I * (1 + Rs / Rsh)
    second_term = V / Rsh
    third_term = I0_val * (np.exp((V + I * Rs) / (n * N * vth)) - 1)
    return first_term + second_term + third_term


# ============================================================================
# CORE GROUPING FUNCTIONS
# ============================================================================

def group_modules_by_kmeans(panel_currents: Dict[str, float], 
                           n_orientations: int,
                           reporter_ids: List[str]) -> Tuple[Dict[str, int], int, List[Tuple[float, float]]]:
    """
    Group modules by K-means clustering based on panel current values for multi-orientation analysis.
    
    Args:
        panel_currents: Dictionary mapping reporter_id to panel_current
        n_orientations: Number of clusters (k) for K-means
        reporter_ids: List of reporter IDs
        
    Returns:
        Tuple of (group_assignments, effective_groups, group_ranges)
        - group_assignments: Dict mapping reporter_id to group number (1-indexed)
        - effective_groups: Actual number of groups created
        - group_ranges: List of (min_current, max_current) for each group
    """
    # Filter out invalid current values
    valid_currents = {}
    for reporter_id in reporter_ids:
        current = panel_currents.get(reporter_id, 0)
        if not (np.isnan(current) or current <= 0):
            valid_currents[reporter_id] = current
    
    if len(valid_currents) < 2:
        # Not enough valid data for clustering
        group_assignments = {rid: 1 for rid in reporter_ids}
        return group_assignments, 1, [(0, max(panel_currents.values()) if panel_currents else 1)]
    
    # Check if we have fewer data points than desired clusters
    if len(valid_currents) < n_orientations:
        warnings.warn(f"Only {len(valid_currents)} valid data points, reducing clusters to {len(valid_currents)}")
        n_clusters = len(valid_currents)
    else:
        n_clusters = n_orientations
    
    # Create DataFrame for easier processing
    current_df = pd.DataFrame(list(valid_currents.items()), columns=['reporter_id', 'current'])
    
    # Check if all values are identical (would cause K-means to fail)
    if current_df['current'].nunique() == 1:
        # All currents are identical, assign all to group 1
        group_assignments = {rid: 1 for rid in reporter_ids}
        unique_current = current_df['current'].iloc[0]
        return group_assignments, 1, [(unique_current, unique_current)]
    
    try:
        # Prepare data for K-means (reshape to 2D array)
        X = current_df['current'].values.reshape(-1, 1)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate mean current for each cluster to ensure consistent ordering
        cluster_means = []
        for cluster_id in range(n_clusters):
            cluster_mask = (cluster_labels == cluster_id)
            if np.any(cluster_mask):
                cluster_mean = current_df[cluster_mask]['current'].mean()
                cluster_means.append((cluster_id, cluster_mean))
        
        # Sort clusters by mean current (ascending order)
        cluster_means.sort(key=lambda x: x[1])
        
        # Create mapping from original cluster ID to ordered group number
        cluster_to_group = {}
        for new_group_id, (original_cluster_id, _) in enumerate(cluster_means):
            cluster_to_group[original_cluster_id] = new_group_id + 1  # 1-indexed groups
        
        # Apply consistent group ordering (Group 1 = lowest current, Group 2 = second lowest, etc.)
        current_df['group'] = [cluster_to_group[label] for label in cluster_labels]
        effective_groups = n_clusters
        
    except Exception as e:
        # Fallback to equal-count ranking if K-means fails
        warnings.warn(f"K-means clustering failed ({str(e)}), using rank-based grouping with {len(valid_currents)} modules")
        current_df = current_df.sort_values('current')
        group_size = len(current_df) // n_clusters
        remainder = len(current_df) % n_clusters
        
        group_assignments_list = []
        for i in range(n_clusters):
            size = group_size + (1 if i < remainder else 0)
            group_assignments_list.extend([i + 1] * size)
        
        current_df['group'] = group_assignments_list
        effective_groups = n_clusters
    
    # Create group assignments dictionary
    group_assignments = {}
    for _, row in current_df.iterrows():
        group_assignments[row['reporter_id']] = int(row['group'])
    
    # Assign remaining (invalid) modules to group 1
    for reporter_id in reporter_ids:
        if reporter_id not in group_assignments:
            group_assignments[reporter_id] = 1
    
    # Calculate group ranges
    group_ranges = []
    for group_num in range(1, effective_groups + 1):
        group_currents = current_df[current_df['group'] == group_num]['current']
        if len(group_currents) > 0:
            group_ranges.append((group_currents.min(), group_currents.max()))
        else:
            group_ranges.append((0, 0))
    
    return group_assignments, effective_groups, group_ranges


def determine_consistent_groups(grouping_history: List[Dict[str, int]], 
                               reporter_ids: List[str]) -> Dict[str, int]:
    """
    Analyze time series of group assignments to determine consistent group for each module.
    
    Args:
        grouping_history: List of group assignment dictionaries from each timestamp
        reporter_ids: List of reporter IDs
        
    Returns:
        Dictionary mapping reporter_id to consistent_group_number based on most frequent assignment
    """
    from collections import Counter, defaultdict
    
    # Track group assignments for each reporter across all timestamps
    reporter_group_counts = defaultdict(Counter)
    
    # Collect group assignments across all timestamps
    for group_assignments in grouping_history:
        for reporter_id, group_id in group_assignments.items():
            reporter_group_counts[reporter_id][group_id] += 1
    
    # Determine most frequent group for each reporter
    consistent_groups = {}
    for reporter_id in reporter_ids:
        if reporter_id in reporter_group_counts and reporter_group_counts[reporter_id]:
            # Find the most common group assignment
            most_common_group = reporter_group_counts[reporter_id].most_common(1)[0][0]
            consistent_groups[reporter_id] = most_common_group
        else:
            # Fallback: assign to group 1 if no data
            consistent_groups[reporter_id] = 1
    
    return consistent_groups


def calculate_multi_string_series_power(grouped_modules: Dict[int, List[str]],
                                      merged_data: pd.DataFrame,
                                      timestamp_idx: int,
                                      currents: np.ndarray,
                                      module_params: Dict) -> Tuple[float, Dict[int, float], bool]:
    """
    Calculate series power for each group, then sum across groups.
    
    Args:
        grouped_modules: Dictionary mapping group_id to list of reporter_ids
        merged_data: DataFrame with timestamp data
        timestamp_idx: Index of current timestamp
        currents: Array of current values for I-V calculation
        module_params: Module parameters (Rs, Rsh, n, N, etc.)
        
    Returns:
        Tuple of (total_multi_string_power, group_powers, valid_calculation)
    """
    group_powers = {}
    valid_groups = 0
    
    for group_id, reporter_list in grouped_modules.items():
        if not reporter_list:
            group_powers[group_id] = 0.0
            continue
            
        # Calculate series voltage for this group
        group_voltage = np.zeros_like(currents)
        group_has_valid_data = False
        
        for reporter_id in reporter_list:
            voltage_col = f'panel_voltage_{reporter_id}'
            current_col = f'panel_current_{reporter_id}'
            temp_col = f'panel_temperature_{reporter_id}'
            
            if not all(col in merged_data.columns for col in [voltage_col, current_col, temp_col]):
                continue
                
            panel_voltage = merged_data[voltage_col].iloc[timestamp_idx]
            panel_current = merged_data[current_col].iloc[timestamp_idx]
            panel_temperature = merged_data[temp_col].iloc[timestamp_idx]
            
            # Skip invalid data
            if (panel_voltage == 0 or panel_current == 0 or 
                np.isnan(panel_voltage) or np.isnan(panel_current)):
                continue
                
            group_has_valid_data = True
            
            # Calculate I-V curve for this module
            panel_temperature_kelvin = panel_temperature + 273.15
            vth = (BOLTZMANN_CONSTANT * panel_temperature_kelvin / ELECTRON_CHARGE 
                   if USE_DYNAMIC_VTH else 0.0259)  # 25°C thermal voltage
            
            # Calculate single-diode parameters
            I0_val = I0(panel_current, panel_voltage, 
                       module_params['Rs'], module_params['Rsh'], 
                       module_params['n'], module_params['N'], vth)
            IL_val = IL(panel_current, panel_voltage,
                       module_params['Rs'], module_params['Rsh'], 
                       module_params['n'], module_params['N'], vth, I0_val)
            
            # Generate I-V curve using pvlib
            params = {
                'photocurrent': IL_val,
                'saturation_current': I0_val,
                'resistance_series': module_params['Rs'],
                'resistance_shunt': module_params['Rsh'],
                'nNsVth': module_params['n'] * module_params['N'] * vth
            }
            
            voltage = pvlib.pvsystem.v_from_i(
                current=currents,
                photocurrent=params['photocurrent'],
                saturation_current=params['saturation_current'],
                resistance_series=params['resistance_series'],
                resistance_shunt=params['resistance_shunt'],
                nNsVth=params['nNsVth']
            )
            
            # Clip voltages where current exceeds Isc
            results = pvlib.pvsystem.singlediode(**params)
            isc = results['i_sc']
            voltage = np.where(currents > isc, 0, voltage)
            
            # Add to group voltage (series connection)
            group_voltage += voltage
        
        if group_has_valid_data:
            # Calculate group power and find maximum
            group_power = group_voltage * currents
            max_group_power = np.max(group_power)
            group_powers[group_id] = max_group_power
            valid_groups += 1
        else:
            group_powers[group_id] = 0.0
    
    # Sum all group powers
    total_multi_string_power = sum(group_powers.values())
    valid_calculation = valid_groups > 0
    
    return total_multi_string_power, group_powers, valid_calculation


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_color_coded_plots(group_assignments: Dict[str, int], 
                            n_groups: int, 
                            merged_data: pd.DataFrame,
                            timestamp_idx: int,
                            reporter_ids: List[str],
                            axs_long: np.ndarray,
                            currents: np.ndarray,
                            module_params: Dict,
                            group_ranges: List[Tuple[float, float]]) -> Tuple[Dict[str, str], List, List]:
    """
    Generate color-coded visualizations showing module groups with enhanced multi-string plotting.
    
    Args:
        group_assignments: Dictionary mapping reporter_id to group
        n_groups: Number of groups
        merged_data: DataFrame with data
        timestamp_idx: Current timestamp index
        reporter_ids: List of reporter IDs
        axs_long: Matplotlib axes for plotting
        currents: Current array for I-V curves
        module_params: Module parameters
        group_ranges: Current ranges for each group
        
    Returns:
        Tuple of (group_colors, legend_handles, legend_labels)
        - group_colors: Dictionary mapping group_id to color
        - legend_handles: List of legend handles for plot (b)
        - legend_labels: List of legend labels for plot (b)
    """
    # Create viridis colormap for groups
    colors = plt.cm.viridis(np.linspace(0, 1, max(n_groups, 3)))
    group_colors = {i+1: colors[i] for i in range(n_groups)}
    
    # Group modules by assignment
    grouped_modules = {}
    for reporter_id, group_id in group_assignments.items():
        if group_id not in grouped_modules:
            grouped_modules[group_id] = []
        grouped_modules[group_id].append(reporter_id)
    
    # Plot raw MPP values (colored by group)
    plotted_groups = set()
    for group_id, reporter_list in grouped_modules.items():
        color = group_colors[group_id]
        for reporter_id in reporter_list:
            voltage_col = f'panel_voltage_{reporter_id}'
            current_col = f'panel_current_{reporter_id}'
            
            if voltage_col in merged_data.columns and current_col in merged_data.columns:
                panel_voltage = merged_data[voltage_col].iloc[timestamp_idx]
                panel_current = merged_data[current_col].iloc[timestamp_idx]
                
                if not (panel_voltage == 0 or panel_current == 0 or 
                       np.isnan(panel_voltage) or np.isnan(panel_current)):
                    label = f'Group {group_id}' if group_id not in plotted_groups else ""
                    axs_long[0].plot(panel_voltage, panel_current, 'ro', 
                                   markersize=6, alpha=0.8, label=label)
                    plotted_groups.add(group_id)
                else:
                    axs_long[0].plot(0, 0, 'kx', alpha=0.5)
    
    # Plot reconstructed I-V curves (colored by group)  
    plotted_groups_iv = set()
    for group_id, reporter_list in grouped_modules.items():
        color = group_colors[group_id]
        for reporter_id in reporter_list:
            voltage_col = f'panel_voltage_{reporter_id}'
            current_col = f'panel_current_{reporter_id}'
            temp_col = f'panel_temperature_{reporter_id}'
            
            if not all(col in merged_data.columns for col in [voltage_col, current_col, temp_col]):
                continue
                
            panel_voltage = merged_data[voltage_col].iloc[timestamp_idx]
            panel_current = merged_data[current_col].iloc[timestamp_idx]
            panel_temperature = merged_data[temp_col].iloc[timestamp_idx]
            
            if (panel_voltage == 0 or panel_current == 0 or 
                np.isnan(panel_voltage) or np.isnan(panel_current)):
                # Plot zero curve
                voltage = np.zeros_like(currents)
                axs_long[1].plot(voltage, currents, '--', color='gray', alpha=0.3)
                continue
            
            # Calculate I-V curve
            panel_temperature_kelvin = panel_temperature + 273.15
            vth = (BOLTZMANN_CONSTANT * panel_temperature_kelvin / ELECTRON_CHARGE 
                   if USE_DYNAMIC_VTH else 0.0259)
            
            I0_val = I0(panel_current, panel_voltage, 
                       module_params['Rs'], module_params['Rsh'], 
                       module_params['n'], module_params['N'], vth)
            IL_val = IL(panel_current, panel_voltage,
                       module_params['Rs'], module_params['Rsh'], 
                       module_params['n'], module_params['N'], vth, I0_val)
            
            params = {
                'photocurrent': IL_val,
                'saturation_current': I0_val,
                'resistance_series': module_params['Rs'],
                'resistance_shunt': module_params['Rsh'],
                'nNsVth': module_params['n'] * module_params['N'] * vth
            }
            
            voltage = pvlib.pvsystem.v_from_i(
                current=currents,
                photocurrent=params['photocurrent'],
                saturation_current=params['saturation_current'],
                resistance_series=params['resistance_series'],
                resistance_shunt=params['resistance_shunt'],
                nNsVth=params['nNsVth']
            )
            
            results = pvlib.pvsystem.singlediode(**params)
            isc = results['i_sc']
            voltage = np.where(currents > isc, 0, voltage)
            
            label = f'Group {group_id}' if group_id not in plotted_groups_iv else ""
            axs_long[1].plot(voltage, currents, color=color, alpha=0.7, linewidth=2, label=label)
            axs_long[1].plot(panel_voltage, panel_current, 'ro', markersize=4)
            plotted_groups_iv.add(group_id)
    
    # Legend for plot (b) will be created outside the plot in the main plotting function
    # Store legend information for later use
    legend_handles = []
    legend_labels = []
    if plotted_groups_iv:
        for group_id in sorted(plotted_groups_iv):
            # Create dummy legend entries for the groups
            legend_handles.append(plt.Line2D([0], [0], color=group_colors[group_id], linewidth=2))
            legend_labels.append(f'Group {group_id}')
    
    return group_colors, legend_handles, legend_labels


def plot_multi_string_iv_curves(grouped_modules: Dict[int, List[str]],
                               merged_data: pd.DataFrame,
                               timestamp_idx: int,
                               currents: np.ndarray,
                               module_params: Dict,
                               axs_long: np.ndarray,
                               group_colors: Dict[str, str]) -> None:
    """
    Plot individual string I-V curves for each group with their MPPs on the right subplot.
    
    Args:
        grouped_modules: Dictionary mapping group_id to list of reporter_ids
        merged_data: DataFrame with timestamp data
        timestamp_idx: Current timestamp index
        currents: Array of current values for I-V calculation
        module_params: Module parameters
        axs_long: Matplotlib axes array
        group_colors: Color mapping for groups
    """
    legend_entries = []
    
    for group_id, reporter_list in grouped_modules.items():
        if not reporter_list:
            continue
            
        color = group_colors[group_id]
        
        # Calculate series voltage for this group
        group_voltage = np.zeros_like(currents)
        group_has_valid_data = False
        
        for reporter_id in reporter_list:
            voltage_col = f'panel_voltage_{reporter_id}'
            current_col = f'panel_current_{reporter_id}'
            temp_col = f'panel_temperature_{reporter_id}'
            
            if not all(col in merged_data.columns for col in [voltage_col, current_col, temp_col]):
                continue
                
            panel_voltage = merged_data[voltage_col].iloc[timestamp_idx]
            panel_current = merged_data[current_col].iloc[timestamp_idx]
            panel_temperature = merged_data[temp_col].iloc[timestamp_idx]
            
            # Skip invalid data
            if (panel_voltage == 0 or panel_current == 0 or 
                np.isnan(panel_voltage) or np.isnan(panel_current)):
                continue
                
            group_has_valid_data = True
            
            # Calculate I-V curve for this module
            panel_temperature_kelvin = panel_temperature + 273.15
            vth = (BOLTZMANN_CONSTANT * panel_temperature_kelvin / ELECTRON_CHARGE 
                   if USE_DYNAMIC_VTH else 0.0259)
            
            # Calculate single-diode parameters
            I0_val = I0(panel_current, panel_voltage, 
                       module_params['Rs'], module_params['Rsh'], 
                       module_params['n'], module_params['N'], vth)
            IL_val = IL(panel_current, panel_voltage,
                       module_params['Rs'], module_params['Rsh'], 
                       module_params['n'], module_params['N'], vth, I0_val)
            
            # Generate I-V curve using pvlib
            params = {
                'photocurrent': IL_val,
                'saturation_current': I0_val,
                'resistance_series': module_params['Rs'],
                'resistance_shunt': module_params['Rsh'],
                'nNsVth': module_params['n'] * module_params['N'] * vth
            }
            
            voltage = pvlib.pvsystem.v_from_i(
                current=currents,
                photocurrent=params['photocurrent'],
                saturation_current=params['saturation_current'],
                resistance_series=params['resistance_series'],
                resistance_shunt=params['resistance_shunt'],
                nNsVth=params['nNsVth']
            )
            
            # Clip voltages where current exceeds Isc
            results = pvlib.pvsystem.singlediode(**params)
            isc = results['i_sc']
            voltage = np.where(currents > isc, 0, voltage)
            
            # Add to group voltage (series connection)
            group_voltage += voltage
        
        if group_has_valid_data:
            # Calculate group power and find maximum
            group_power = group_voltage * currents
            max_power_idx = np.argmax(group_power)
            max_group_voltage = group_voltage[max_power_idx]
            max_group_current = currents[max_power_idx]
            max_group_power = group_power[max_power_idx]
            
            # Plot string I-V curve
            axs_long[2].plot(group_voltage, currents, color=color, linewidth=2, alpha=0.8,
                           label=f'String {group_id} IV')
            
            # Plot string MPP
            axs_long[2].plot(max_group_voltage, max_group_current, 'ro', markersize=8, alpha=0.9,
                           label=f'Multi String MPP {group_id}')
            
            legend_entries.append(f'String {group_id} IV')
            legend_entries.append(f'Multi String MPP {group_id}')
    
    # Plot (c) will have no legend per user requirements


def create_raw_data_plots(merged_data: pd.DataFrame, 
                         reporter_ids: List[str],
                         site_id: str,
                         season: str,
                         output_dir: str,
                         num_days_to_plot: int = 10) -> None:
    """
    Create 2x2 subplot showing raw sensor data over time.
    
    Args:
        merged_data: DataFrame with all sensor data
        reporter_ids: List of reporter IDs
        site_id: Site identifier  
        season: Season name
        output_dir: Directory to save plot
        num_days_to_plot: Number of days to include in plot
    """
    # Filter data for specified number of days
    start_date = pd.to_datetime(merged_data['Timestamp'].iloc[0])
    end_date = start_date + timedelta(days=num_days_to_plot)
    filtered_data = merged_data[
        (pd.to_datetime(merged_data['Timestamp']) >= start_date) & 
        (pd.to_datetime(merged_data['Timestamp']) < end_date)
    ]
    
    first_month = pd.to_datetime(merged_data['Timestamp'].iloc[0]).strftime('%B')
    
    # Create 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=LONG_HOZ_FIGSIZE)
    
    # Add figure-level title
    fig.suptitle(f"Site ID: {site_id} | Month: {first_month}", fontsize=TITLE_SIZE)
    
    # Plot all panel currents
    for reporter_id in reporter_ids:
        axs[0, 0].plot(filtered_data['Timestamp'], filtered_data[f'panel_current_{reporter_id}'])
    axs[0, 0].set_title('Panel Current', fontsize=TITLE_SIZE)
    axs[0, 0].set_xlabel('Time', fontsize=AXIS_LABEL_SIZE-5)
    axs[0, 0].set_ylabel('Current (A)', fontsize=AXIS_LABEL_SIZE-5)
    axs[0, 0].tick_params(axis='x', which='both', bottom=False, labelbottom=False, labelsize=AXIS_NUM_SIZE)
    axs[0, 0].tick_params(axis='y', labelsize=AXIS_NUM_SIZE)
    
    # Plot all panel voltages
    for reporter_id in reporter_ids:
        axs[0, 1].plot(filtered_data['Timestamp'], filtered_data[f'panel_voltage_{reporter_id}'])
    axs[0, 1].set_title('Panel Voltage', fontsize=TITLE_SIZE)
    axs[0, 1].set_xlabel('Time', fontsize=AXIS_LABEL_SIZE-5)
    axs[0, 1].set_ylabel('Voltage (V)', fontsize=AXIS_LABEL_SIZE-5)
    axs[0, 1].tick_params(axis='x', which='both', bottom=False, labelbottom=False, labelsize=AXIS_NUM_SIZE)
    axs[0, 1].tick_params(axis='y', labelsize=AXIS_NUM_SIZE)
    
    # Plot all temperatures
    for reporter_id in reporter_ids:
        axs[1, 0].plot(filtered_data['Timestamp'], filtered_data[f'temperature_{reporter_id}'])
    axs[1, 0].set_title('Temperature', fontsize=TITLE_SIZE)
    axs[1, 0].set_xlabel('Time', fontsize=AXIS_LABEL_SIZE-5)
    axs[1, 0].set_ylabel('Temperature (°C)', fontsize=AXIS_LABEL_SIZE-5)
    axs[1, 0].tick_params(axis='x', which='both', bottom=False, labelbottom=False, labelsize=AXIS_NUM_SIZE)
    axs[1, 0].tick_params(axis='y', labelsize=AXIS_NUM_SIZE)
    
    # Plot all panel temperatures
    for reporter_id in reporter_ids:
        axs[1, 1].plot(filtered_data['Timestamp'], filtered_data[f'panel_temperature_{reporter_id}'])
    axs[1, 1].set_title('Panel Temperature', fontsize=TITLE_SIZE)
    axs[1, 1].set_xlabel('Time', fontsize=AXIS_LABEL_SIZE-5)
    axs[1, 1].set_ylabel('Panel Temperature (°C)', fontsize=AXIS_LABEL_SIZE-5)
    axs[1, 1].tick_params(axis='x', which='both', bottom=False, labelbottom=False, labelsize=AXIS_NUM_SIZE)
    axs[1, 1].tick_params(axis='y', labelsize=AXIS_NUM_SIZE)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle
    
    # Save the figure
    plot_path = os.path.join(output_dir, f"{site_id}_{season}_data.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close to free memory
    
    print(f"Raw data plot saved: {plot_path}")


# def create_power_comparison_plot(iv_sum_data: pd.DataFrame,
#                                 pmppt_data: pd.DataFrame,
#                                 site_id: str,
#                                 season: str,
#                                 output_dir: str) -> None:
#     """
#     Create power comparison plot showing Series connection vs Sum of maximum powers.
    
#     Args:
#         iv_sum_data: DataFrame with sum of I*V power data
#         pmppt_data: DataFrame with series connection power data  
#         site_id: Site identifier
#         season: Season name
#         output_dir: Directory to save plot
#     """
#     # Combine the data
#     combined_data = pd.merge(iv_sum_data, pmppt_data['Pmppt (W)'], left_index=True, right_index=True, how='outer')
    
#     # Ensure timestamp is in datetime format
#     combined_data['Timestamp'] = pd.to_datetime(combined_data['Timestamp'])
    
#     # Calculate overall mismatch
#     sum_iv_E = combined_data['Sum of I*V (W)'].sum()
#     pmppt_E = combined_data['Pmppt (W)'].sum()
#     sum_mismatch = (sum_iv_E - pmppt_E) / sum_iv_E if sum_iv_E > 0 else 0
    
#     # Create the plot
#     fig, ax = plt.subplots(figsize=LONG_HOZ_FIGSIZE)
    
#     # Plot both power curves
#     ax.plot(combined_data['Timestamp'],
#             combined_data['Pmppt (W)'],
#             label='Series connection',
#             alpha=0.4)
#     ax.plot(combined_data['Timestamp'],
#             combined_data['Sum of I*V (W)'],
#             label='Sum of maximum powers',
#             alpha=0.4)
    
#     # Set labels and title
#     ax.set_xlabel('Time', fontsize=AXIS_LABEL_SIZE)
#     ax.set_ylabel('Power (W)', fontsize=AXIS_LABEL_SIZE)
    
#     first_month = pd.to_datetime(combined_data['Timestamp'].iloc[0]).strftime('%B')
#     ax.set_title(
#         f'Site ID: {site_id}, Month: {first_month}\nMismatch: {sum_mismatch * 100:.2f}%',
#         fontsize=TITLE_SIZE, pad=20
#     )
    
#     # Add legend
#     ax.legend(loc='upper right', fontsize=AXIS_NUM_SIZE-5)
    
#     # Format x-axis ticks every 2 days
#     ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
#     ax.tick_params(axis='x', labelsize=AXIS_NUM_SIZE)
#     ax.tick_params(axis='y', labelsize=AXIS_NUM_SIZE)
    
#     # Adjust layout
#     plt.tight_layout()
#     fig.subplots_adjust(bottom=0.20)
    
#     # Save the plot
#     plot_path = os.path.join(output_dir, 'pmppt_vs_sum_iv.png')
#     fig.savefig(plot_path, dpi=300)
#     plt.close(fig)  # Close to free memory
    
#     print(f"Power comparison plot saved: {plot_path}")
    
#     return sum_mismatch


def create_power_comparison_plot(iv_sum_data: pd.DataFrame,
                               pmppt_data: pd.DataFrame,
                               multi_string_data: pd.DataFrame,
                               site_id: str,
                               season: str,
                               output_dir: str) -> Tuple[float, float]:
    """
    Create power comparison plot showing Series connection vs Multi-string vs Sum of maximum powers.
    
    Args:
        iv_sum_data: DataFrame with sum of I*V power data
        pmppt_data: DataFrame with series connection power data
        multi_string_data: DataFrame with multi-string power data
        site_id: Site identifier
        season: Season name
        output_dir: Directory to save plot
        
    Returns:
        Tuple of (traditional_mismatch, multi_string_mismatch)
    """
    # Combine the data using timestamp-based merging for proper alignment
    combined_data = pd.merge(iv_sum_data, pmppt_data[['Timestamp', 'Pmppt (W)']], on='Timestamp', how='outer')
    combined_data = pd.merge(combined_data, multi_string_data[['Timestamp', 'Multi_String_Power (W)']], 
                            on='Timestamp', how='outer')
    
    # Ensure timestamp is in datetime format
    combined_data['Timestamp'] = pd.to_datetime(combined_data['Timestamp'])
    
    # Calculate overall mismatches
    sum_iv_E = combined_data['Sum of I*V (W)'].sum()
    pmppt_E = combined_data['Pmppt (W)'].sum()
    multi_string_E = combined_data['Multi_String_Power (W)'].sum()
    
    trad_mismatch = (sum_iv_E - pmppt_E) / sum_iv_E if sum_iv_E > 0 else 0
    multi_mismatch = (sum_iv_E - multi_string_E) / sum_iv_E if sum_iv_E > 0 else 0
    improvement = trad_mismatch - multi_mismatch
    
    # Create the plot
    fig, ax = plt.subplots(figsize=LONG_HOZ_FIGSIZE)
    
    # Plot all power curves
    ax.plot(combined_data['Timestamp'],
            combined_data['Pmppt (W)'],
            label='Series connection',
            alpha=0.4)
    ax.plot(combined_data['Timestamp'],
            combined_data['Multi_String_Power (W)'],
            label='Consistent Multi-string connection',
            alpha=0.4, 
            linestyle='--')
    ax.plot(combined_data['Timestamp'],
            combined_data['Sum of I*V (W)'],
            label='Sum of maximum powers',
            alpha=0.4)
    
    # Set labels and title
    ax.set_xlabel('Time', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('Power (W)', fontsize=AXIS_LABEL_SIZE)
    
    first_month = pd.to_datetime(combined_data['Timestamp'].iloc[0]).strftime('%B')
    ax.set_title(
        f'Site ID: {site_id}, Month: {first_month}\n'
        f'Traditional Mismatch: {trad_mismatch * 100:.2f}% | '
        f'Consistent Multi-String Mismatch: {multi_mismatch * 100:.2f}% | '
        f'Improvement: {improvement * 100:.2f}%',
        fontsize=TITLE_SIZE, pad=20
    )
    
    # Add legend
    ax.legend(loc='upper right', fontsize=AXIS_NUM_SIZE-5)
    
    # Format x-axis ticks every 2 days
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.tick_params(axis='x', labelsize=AXIS_NUM_SIZE)
    ax.tick_params(axis='y', labelsize=AXIS_NUM_SIZE)
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'power_comparison_all_methods.png')
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)  # Close to free memory
    
    print(f"Power comparison plot saved: {plot_path}")
    print(f"Traditional mismatch: {trad_mismatch * 100:.2f}%")
    print(f"Consistent multi-string mismatch: {multi_mismatch * 100:.2f}%")
    print(f"Improvement: {improvement * 100:.2f}%")
    
    return trad_mismatch, multi_mismatch


def create_simplified_power_comparison_plot(iv_sum_data: pd.DataFrame,
                                           multi_string_data: pd.DataFrame,
                                           site_id: str,
                                           season: str,
                                           output_dir: str) -> float:
    """
    Create simplified power comparison plot showing Sum of maximum powers vs Multi-string consistent grouping only.
    
    Args:
        iv_sum_data: DataFrame with sum of I*V power data
        multi_string_data: DataFrame with multi-string power data
        site_id: Site identifier
        season: Season name
        output_dir: Directory to save plot
        
    Returns:
        Multi-string mismatch percentage
    """
    # Combine the data using timestamp-based merging for proper alignment
    combined_data = pd.merge(iv_sum_data, multi_string_data[['Timestamp', 'Multi_String_Power (W)']], 
                            on='Timestamp', how='outer')
    
    # Ensure timestamp is in datetime format
    combined_data['Timestamp'] = pd.to_datetime(combined_data['Timestamp'])
    
    # Calculate mismatch
    sum_iv_E = combined_data['Sum of I*V (W)'].sum()
    multi_string_E = combined_data['Multi_String_Power (W)'].sum()
    multi_mismatch = (sum_iv_E - multi_string_E) / sum_iv_E if sum_iv_E > 0 else 0
    
    # Create the plot
    fig, ax = plt.subplots(figsize=LONG_HOZ_FIGSIZE)
    
    # Plot power curves
    ax.plot(combined_data['Timestamp'],
            combined_data['Sum of I*V (W)'],
            label='Sum of maximum powers (Ideal)',
            alpha=0.7,
            linewidth=2)
    ax.plot(combined_data['Timestamp'],
            combined_data['Multi_String_Power (W)'],
            label='Consistent Multi-string connection',
            alpha=0.7, 
            linewidth=2,
            linestyle='--')
    
    # Set labels and title
    ax.set_xlabel('Time', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('Power (W)', fontsize=AXIS_LABEL_SIZE)
    
    first_month = pd.to_datetime(combined_data['Timestamp'].iloc[0]).strftime('%B')
    ax.set_title(
        f'Site ID: {site_id}, Month: {first_month}\n'
        f'Multi-String Mismatch Loss: {multi_mismatch * 100:.2f}%\n'
        f'Consistent Multi-String vs Ideal Comparison',
        fontsize=TITLE_SIZE, pad=20
    )
    
    # Add legend
    ax.legend(loc='upper right', fontsize=AXIS_NUM_SIZE-2)
    
    # Format x-axis ticks every 2 days
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.tick_params(axis='x', labelsize=AXIS_NUM_SIZE)
    ax.tick_params(axis='y', labelsize=AXIS_NUM_SIZE)
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'simplified_power_comparison.png')
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)  # Close to free memory
    
    print(f"Simplified power comparison plot saved: {plot_path}")
    print(f"Multi-string mismatch loss: {multi_mismatch * 100:.2f}%")
    
    return multi_mismatch


# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def load_and_process_site_data(site_id: str, season: str) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Load and process data for a specific site and season.
    
    Returns:
        Tuple of (merged_data, reporter_ids, site_directory)
    """
    # Find site directory
    site_folders = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    matching_dirs = [d for d in site_folders if site_id in d]
    
    if not matching_dirs:
        raise ValueError(f"No directory found for site_id {site_id}")
    
    site_dir = os.path.join(DATA_DIR, matching_dirs[0])
    
    # Load summary data for hemisphere detection
    summary_df = pd.read_excel(SUMMARY_DIR, sheet_name='Sheet1')
    site_info = summary_df[summary_df['Site ID'] == int(site_id)]
    
    if site_info.empty:
        raise ValueError(f"Site {site_id} not found in summary file")
    
    # Determine hemisphere and season mapping
    country = site_info['Country'].values[0]
    season_months = SEASON_MONTHS_SOUTH if country == 'Australia' else SEASON_MONTHS_NORTH
    
    # Find season directory
    season_lower = season.lower()
    season_dir_candidates = [
        d for d in os.listdir(site_dir)
        if (season_lower in d.lower() or 
            any(month in d.lower() for month in season_months.get(season_lower, [])))
    ]
    
    if not season_dir_candidates:
        raise ValueError(f"No folder found for season {season} in site {site_id}")
    
    season_dir = os.path.join(site_dir, season_dir_candidates[0])
    
    # Load optimizer data
    csv_files = [f for f in os.listdir(season_dir) if 'optimizer_data' in f and f.endswith('.csv')]
    
    if not csv_files:
        raise ValueError(f"No optimizer_data CSV files found in {season_dir}")
    
    dataframes = []
    reporter_ids = []
    
    # Process CSV files
    if len(csv_files) == 1:
        # Single file case
        file_path = os.path.join(season_dir, csv_files[0])
        df = pd.read_csv(file_path)
        
        if 'reporter_id' in df.columns:
            # Split by reporter_id
            unique_reporters = df['reporter_id'].unique()
            for reporter in unique_reporters:
                df_rep = df[df['reporter_id'] == reporter].copy()
                if df_rep.columns[0] != 'Timestamp':
                    df_rep.rename(columns={df_rep.columns[0]: 'Timestamp'}, inplace=True)
                
                rename_map = {
                    'panel_current': f'panel_current_{reporter}',
                    'panel_voltage': f'panel_voltage_{reporter}',
                    'temperature': f'temperature_{reporter}',
                    'panel_temperature': f'panel_temperature_{reporter}',
                    'power': f'power_{reporter}'
                }
                df_rep.rename(columns=rename_map, inplace=True)
                
                # Parse timestamp
                for fmt in TIMESTAMP_FORMATS:
                    try:
                        df_rep['Timestamp'] = pd.to_datetime(df_rep['Timestamp'], format=fmt)
                        break
                    except (ValueError, TypeError):
                        pass
                
                df_rep.set_index('Timestamp', inplace=True)
                df_rep = df_rep[list(rename_map.values())]
                dataframes.append(df_rep)
                reporter_ids.append(str(reporter))
        else:
            # Single file without reporter_id
            default_reporter = "default"
            if df.columns[0] != 'Timestamp':
                df.rename(columns={df.columns[0]: 'Timestamp'}, inplace=True)
                
            rename_map = {
                'panel_current': f'panel_current_{default_reporter}',
                'panel_voltage': f'panel_voltage_{default_reporter}',
                'temperature': f'temperature_{default_reporter}',
                'panel_temperature': f'panel_temperature_{default_reporter}',
                'power': f'power_{default_reporter}'
            }
            df.rename(columns=rename_map, inplace=True)
            
            for fmt in TIMESTAMP_FORMATS:
                try:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format=fmt)
                    break
                except (ValueError, TypeError):
                    pass
                    
            df.set_index('Timestamp', inplace=True)
            df = df[list(rename_map.values())]
            dataframes.append(df)
            reporter_ids.append(default_reporter)
    else:
        # Multiple files case
        for file in csv_files:
            file_path = os.path.join(season_dir, file)
            optimizer_data = pd.read_csv(file_path)
            
            reporter_id = file.split('_')[-1].split('.')[0]
            reporter_ids.append(reporter_id)
            
            optimizer_data.rename(columns={
                'panel_current': f'panel_current_{reporter_id}',
                'panel_voltage': f'panel_voltage_{reporter_id}',
                'temperature': f'temperature_{reporter_id}',
                'panel_temperature': f'panel_temperature_{reporter_id}',
                'power': f'power_{reporter_id}'
            }, inplace=True)
            
            if optimizer_data.columns[0] != 'Timestamp':
                optimizer_data.rename(columns={optimizer_data.columns[0]: 'Timestamp'}, inplace=True)
            
            for fmt in TIMESTAMP_FORMATS:
                try:
                    optimizer_data['Timestamp'] = pd.to_datetime(optimizer_data['Timestamp'], format=fmt)
                    break
                except (ValueError, TypeError):
                    pass
            
            optimizer_data.set_index('Timestamp', inplace=True)
            optimizer_data = optimizer_data[[f'panel_current_{reporter_id}', f'panel_voltage_{reporter_id}',
                                            f'temperature_{reporter_id}', f'panel_temperature_{reporter_id}',
                                            f'power_{reporter_id}']]
            dataframes.append(optimizer_data)
    
    # Synchronize timestamps
    if dataframes:
        earliest_timestamp = max([df.index[0] for df in dataframes])
        latest_timestamp = min([df.index[-1] for df in dataframes])
        new_index = pd.date_range(start=earliest_timestamp, end=latest_timestamp, freq='5min')
        
        for i in range(len(dataframes)):
            for index in new_index:
                if index not in dataframes[i].index:
                    dataframes[i].loc[index] = np.nan
        
        merged_data = pd.concat(dataframes, axis=1)
        merged_data.reset_index(inplace=True)
    else:
        raise ValueError("No valid dataframes created")
    
    # Handle temperature substitution
    if USE_A_T:
        cols_to_drop = [col for col in merged_data.columns if 'panel_temperature' in col]
        merged_data.drop(columns=cols_to_drop, inplace=True)
        
        for col in [col for col in merged_data.columns if 'temperature' in col]:
            new_col = col.replace('temperature', 'panel_temperature')
            merged_data[new_col] = merged_data[col]
    
    return merged_data, reporter_ids, site_dir


def extract_module_parameters(site_dir: str) -> Dict:
    """Extract module parameters from .PAN file."""
    pan_files = [f for f in os.listdir(site_dir) if f.endswith('.PAN')]
    
    if not pan_files:
        raise ValueError(f"No .PAN file found in {site_dir}")
    
    pan_file_path = os.path.join(site_dir, pan_files[0])
    
    params = {}
    with open(pan_file_path, 'r') as f:
        pan_data = f.readlines()
        
        for line in pan_data:
            if 'RSerie' in line:
                params['Rs'] = float(line.split('=')[1].strip())
            elif 'RShunt' in line:
                params['Rsh'] = float(line.split('=')[1].strip())
            elif 'NCelS' in line:
                params['N'] = int(line.split('=')[1].strip())
            elif 'Gamma' in line:
                params['n'] = float(line.split('=')[1].strip())
                break
    
    required_params = ['Rs', 'Rsh', 'N', 'n']
    for param in required_params:
        if param not in params:
            raise ValueError(f"Required parameter {param} not found in .PAN file")
    
    return params


# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def run_multi_orientation_analysis(site_ids: Optional[List[str]] = None, 
                                  seasons: List[str] = ['spring'],
                                  num_days_to_plot: int = 10) -> None:
    """
    Main function executing the enhanced multi-orientation analysis.
    
    Args:
        site_ids: List of site IDs to process. If None, processes all multi-orientation sites.
        seasons: List of seasons to process
        num_days_to_plot: Number of days to include in analysis
    """
    if site_ids is None:
        site_ids = list(MULTI_ORIENTATION_SITES.keys())
    
    # Ensure only multi-orientation sites are processed
    valid_site_ids = [sid for sid in site_ids if sid in MULTI_ORIENTATION_SITES]
    if not valid_site_ids:
        print("No valid multi-orientation sites found in the provided list.")
        return
    
    print(f"Processing {len(valid_site_ids)} multi-orientation sites: {valid_site_ids}")
    
    for site_id in valid_site_ids:
        n_orientations = MULTI_ORIENTATION_SITES[site_id]
        print(f"\n{'='*60}")
        print(f"Processing Site {site_id} with {n_orientations} orientations")
        print(f"{'='*60}")
        
        for season in seasons:
            print(f"\nProcessing season: {season}")
            
            try:
                # Load data
                merged_data, reporter_ids, site_dir = load_and_process_site_data(site_id, season)
                module_params = extract_module_parameters(site_dir)
                
                print(f"Loaded data for {len(reporter_ids)} modules: {reporter_ids}")
                print(f"Module parameters: Rs={module_params['Rs']}, Rsh={module_params['Rsh']}, "
                      f"N={module_params['N']}, n={module_params['n']}")
                
                # Filter data for specified number of days
                start_date = pd.to_datetime(merged_data['Timestamp'].iloc[0])
                end_date = start_date + timedelta(days=num_days_to_plot)
                filtered_data = merged_data[
                    (pd.to_datetime(merged_data['Timestamp']) >= start_date) & 
                    (pd.to_datetime(merged_data['Timestamp']) < end_date)
                ].copy()
                
                first_month = pd.to_datetime(merged_data['Timestamp'].iloc[0]).strftime('%B')
                print(f"Filtered data: {len(filtered_data)} timestamps in {first_month}")
                
                # Create output directory
                date_folder = datetime.datetime.now().strftime("%y_%m_%d_Results")
                v_from_i_combined_dir = r"C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\24_09_24_Solar_Edge\Results\v_from_i_combined"
                date_results_dir = os.path.join(v_from_i_combined_dir, date_folder)
                os.makedirs(date_results_dir, exist_ok=True)
                
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                results_folder = os.path.join(date_results_dir, f"{site_id}_{first_month}_{timestamp_str}")
                os.makedirs(results_folder, exist_ok=True)
                
                # Process timestamps
                process_site_timestamps(filtered_data, reporter_ids, site_id, season, 
                                      n_orientations, module_params, results_folder)
                
                print(f"Analysis complete for site {site_id}, season {season}")
                print(f"Results saved to: {results_folder}")
                
            except Exception as e:
                print(f"Error processing site {site_id}, season {season}: {str(e)}")
                continue


def process_site_timestamps(merged_data: pd.DataFrame,
                          reporter_ids: List[str], 
                          site_id: str,
                          season: str,
                          n_orientations: int,
                          module_params: Dict,
                          output_dir: str) -> None:
    """
    Process all timestamps for a site with multi-orientation grouping.
    
    Args:
        merged_data: DataFrame with all timestamp data
        reporter_ids: List of module reporter IDs  
        site_id: Site identifier
        season: Season name
        n_orientations: Number of orientation groups
        module_params: Module parameters from .PAN file
        output_dir: Output directory for results
    """
    # Initialize data storage
    image_files = []
    max_power_df_combined = pd.DataFrame(columns=[
        'Timestamp', 'Max Voltage (V)', 'Max Current (A)', 'Max Power (W)', 'Voc (V)', 'Isc (A)'
    ])
    pmppt_data = pd.DataFrame(columns=['Timestamp', 'Pmppt (W)'])
    multi_string_data = pd.DataFrame(columns=['Timestamp', 'Multi_String_Power (W)'])
    module_param_df = pd.DataFrame(columns=[
        'Timestamp', 'Optimizer', 'I0', 'Isc', 'Voc', 'FF', 'Pmp', 'Imp', 'Vmp'
    ])
    iv_sum_data = pd.DataFrame(columns=['Timestamp', 'Sum of I*V (W)'])
    grouping_data = pd.DataFrame(columns=[
        'Timestamp', 'N_Groups', 'Group_ID', 'Reporter_IDs', 'Current_Range_Min', 'Current_Range_Max', 'Group_Power'
    ])
    
    # Define current range for I-V calculations
    currents = np.linspace(0, Y_LIMIT_INVERTER[1], 100)
    
    # Generate raw data plot early in the process (like reference notebook)
    try:
        create_raw_data_plots(merged_data, reporter_ids, site_id, season, output_dir)
    except Exception as e:
        print(f"Warning: Could not create raw data plot: {str(e)}")
    
    print(f"Processing {len(merged_data)} timestamps...")
    
    # ============================================================================
    # PHASE 1: DATA COLLECTION - Collect grouping data across all timestamps
    # ============================================================================
    print("Phase 1: Collecting grouping data across all timestamps...")
    grouping_history = []  # Store group assignments for each timestamp
    timestamp_data = []  # Store timestamp-specific data for later visualization
    
    for idx in range(len(merged_data)):
        current_timestamp = pd.to_datetime(merged_data['Timestamp'].iloc[idx])
        
        # Calculate total system power for night-time detection
        total_system_power = 0
        valid_power_readings = 0
        for optimiser in reporter_ids:
            power_val = merged_data.get(f'power_{optimiser}', pd.Series([0]*len(merged_data))).iloc[idx]
            if not pd.isna(power_val):
                total_system_power += max(0, power_val)
                valid_power_readings += 1

        # Skip only if true night conditions (very low system power) or insufficient valid readings
        if total_system_power < 10 or valid_power_readings < len(reporter_ids) * 0.25:
            if idx % 50 == 0:  # Reduce logging frequency
                print(f"  Timestamp {idx}: Night-time or insufficient data (total power: {total_system_power:.1f}W)")
            
            # Store zero values for night-time instead of complete skip
            if total_system_power < 10:
                timestamp_title = current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                
                # Store zero power data for night-time
                iv_sum_row = pd.DataFrame({'Timestamp': [timestamp_title], 'Sum of I*V (W)': [0.0]})
                pmppt_row = pd.DataFrame({'Timestamp': [timestamp_title], 'Pmppt (W)': [0.0]})  
                multi_string_row = pd.DataFrame({'Timestamp': [timestamp_title], 'Multi_String_Power (W)': [0.0]})

                # Append to respective DataFrames
                if iv_sum_data.empty:
                    iv_sum_data = iv_sum_row
                else:
                    iv_sum_data = pd.concat([iv_sum_data, iv_sum_row], ignore_index=True)
                
                if pmppt_data.empty:
                    pmppt_data = pmppt_row
                else:
                    pmppt_data = pd.concat([pmppt_data, pmppt_row], ignore_index=True)
                    
                if multi_string_data.empty:
                    multi_string_data = multi_string_row
                else:
                    multi_string_data = pd.concat([multi_string_data, multi_string_row], ignore_index=True)
            
            continue
        
        # Group modules by current percentiles
        panel_currents = {}
        for reporter_id in reporter_ids:
            current_col = f'panel_current_{reporter_id}'
            if current_col in merged_data.columns:
                panel_currents[reporter_id] = merged_data[current_col].iloc[idx]
        
        group_assignments, effective_groups, group_ranges = group_modules_by_kmeans(
            panel_currents, n_orientations, reporter_ids
        )
        
        # Group modules by assignment
        grouped_modules = {}
        for reporter_id, group_id in group_assignments.items():
            if group_id not in grouped_modules:
                grouped_modules[group_id] = []
            grouped_modules[group_id].append(reporter_id)
        
        # Calculate multi-string power
        multi_string_power, group_powers, valid_calculation = calculate_multi_string_series_power(
            grouped_modules, merged_data, idx, currents, module_params
        )
        
        # Calculate sum_iv for physical constraint validation
        current_sum_iv = sum(merged_data[f'panel_voltage_{opt}'].iloc[idx] * merged_data[f'panel_current_{opt}'].iloc[idx] 
                            for opt in reporter_ids 
                            if f'panel_voltage_{opt}' in merged_data.columns and f'panel_current_{opt}' in merged_data.columns
                            and not (np.isnan(merged_data[f'panel_voltage_{opt}'].iloc[idx]) or np.isnan(merged_data[f'panel_current_{opt}'].iloc[idx])))
        
        # Validate physical constraints
        if multi_string_power > current_sum_iv + 0.01:  # Small numerical tolerance
            print(f"WARNING: Multi-string power ({multi_string_power:.2f}W) > Sum MPP ({current_sum_iv:.2f}W)")
            print(f"  Timestamp: {current_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            multi_string_power = min(multi_string_power, current_sum_iv)  # Cap at physical maximum

        # Calculate realistic mismatch loss for monitoring
        if current_sum_iv > 0:
            mismatch_loss_check = (current_sum_iv - multi_string_power) / current_sum_iv * 100
            
            # Warning for suspiciously low losses during good generation
            if mismatch_loss_check < 0.5 and total_system_power > 500:  # Daytime with good generation
                print(f"INFO: Low mismatch loss during good generation: {mismatch_loss_check:.2f}% at {current_timestamp.strftime('%H:%M:%S')}")
        
        # Store grouping information for later processing
        for group_id, reporter_list in grouped_modules.items():
            if group_id <= len(group_ranges):
                min_current, max_current = group_ranges[group_id - 1]
            else:
                min_current, max_current = 0, 0
                
            grouping_row = pd.DataFrame({
                'Timestamp': [current_timestamp],
                'N_Groups': [effective_groups],
                'Group_ID': [group_id],
                'Reporter_IDs': [';'.join(reporter_list)],
                'Current_Range_Min': [min_current],
                'Current_Range_Max': [max_current],
                'Group_Power': [group_powers.get(group_id, 0)]
            })
            if grouping_data.empty:
                grouping_data = grouping_row
            else:
                grouping_data = pd.concat([grouping_data, grouping_row], ignore_index=True)
        
        # Store grouping history for consistent group determination
        grouping_history.append(group_assignments.copy())
        
        # Calculate traditional series power for comparison (essential calculation for Phase 3)
        traditional_combined_voltage = np.zeros_like(currents)
        traditional_valid_data_found = False
        traditional_sum_iv = 0
        
        for optimiser in reporter_ids:
            optimiser_voltage = merged_data[f'panel_voltage_{optimiser}']
            optimiser_current = merged_data[f'panel_current_{optimiser}']
            panel_temperature = merged_data[f'panel_temperature_{optimiser}']
            
            is_nan_or_zero = (
                optimiser_voltage.iloc[idx] == 0 or
                optimiser_current.iloc[idx] == 0 or
                np.isnan(optimiser_voltage.iloc[idx]) or
                np.isnan(optimiser_current.iloc[idx])
            )
            
            if not is_nan_or_zero:
                traditional_valid_data_found = True
                panel_temperature_kelvin = panel_temperature.iloc[idx] + 273.15
                vth = (BOLTZMANN_CONSTANT * panel_temperature_kelvin / ELECTRON_CHARGE 
                       if USE_DYNAMIC_VTH else 0.0259)
                panel_voltage = optimiser_voltage.iloc[idx]
                panel_current = optimiser_current.iloc[idx]
                traditional_sum_iv += panel_voltage * panel_current
                
                # Calculate single-diode parameters for series combination
                I0_op = I0(panel_current, panel_voltage, module_params['Rs'], module_params['Rsh'], 
                          module_params['n'], module_params['N'], vth)
                IL_op = IL(panel_current, panel_voltage, module_params['Rs'], module_params['Rsh'], 
                          module_params['n'], module_params['N'], vth, I0_op)
                
                params = {
                    'photocurrent': IL_op,
                    'saturation_current': I0_op,
                    'resistance_series': module_params['Rs'],
                    'resistance_shunt': module_params['Rsh'],
                    'nNsVth': module_params['n'] * module_params['N'] * vth
                }
                
                voltage = pvlib.pvsystem.v_from_i(
                    current=currents,
                    photocurrent=params['photocurrent'],
                    saturation_current=params['saturation_current'],
                    resistance_series=params['resistance_series'],
                    resistance_shunt=params['resistance_shunt'],
                    nNsVth=params['nNsVth']
                )
                
                results = pvlib.pvsystem.singlediode(**params)
                isc = results['i_sc']
                voltage = np.where(currents > isc, 0, voltage)
                traditional_combined_voltage += voltage
        
        # Calculate traditional series max power
        traditional_max_power = np.nan
        if traditional_valid_data_found:
            traditional_power = traditional_combined_voltage * currents
            traditional_max_power = np.max(traditional_power)
            
            # Store traditional series results for export
            max_power_idx = np.argmax(traditional_power)
            max_voltage = traditional_combined_voltage[max_power_idx]
            max_current = currents[max_power_idx]
            isc_combined = currents[np.where(traditional_combined_voltage > 0)[0][-1]] if np.any(traditional_combined_voltage > 0) else 0
            voc_combined = traditional_combined_voltage[np.where(currents == 0)[0][0]] if len(np.where(currents == 0)[0]) > 0 else 0
            
            max_power_point = pd.DataFrame({
                'Timestamp': [current_timestamp],
                'Max Voltage (V)': [max_voltage],
                'Max Current (A)': [max_current], 
                'Max Power (W)': [traditional_max_power],
                'Voc (V)': [voc_combined],
                'Isc (A)': [isc_combined]
            })
            if max_power_df_combined.empty:
                max_power_df_combined = max_power_point
            else:
                max_power_df_combined = pd.concat([max_power_df_combined, max_power_point], ignore_index=True)
            
            pmppt_row = pd.DataFrame({
                'Timestamp': [current_timestamp], 'Pmppt (W)': [traditional_max_power]
            })
            if pmppt_data.empty:
                pmppt_data = pmppt_row
            else:
                pmppt_data = pd.concat([pmppt_data, pmppt_row], ignore_index=True)
        
        # Store sum_iv data
        timestamp_title = current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        iv_sum_row = pd.DataFrame({
            'Timestamp': [timestamp_title], 'Sum of I*V (W)': [traditional_sum_iv]
        })
        if iv_sum_data.empty:
            iv_sum_data = iv_sum_row
        else:
            iv_sum_data = pd.concat([iv_sum_data, iv_sum_row], ignore_index=True)
        
        multi_string_row = pd.DataFrame({
            'Timestamp': [timestamp_title], 'Multi_String_Power (W)': [multi_string_power]
        })
        if multi_string_data.empty:
            multi_string_data = multi_string_row
        else:
            multi_string_data = pd.concat([multi_string_data, multi_string_row], ignore_index=True)
        
        # Store timestamp data for later visualization and analysis
        timestamp_info = {
            'idx': idx,
            'timestamp': current_timestamp,
            'group_assignments': group_assignments.copy(),
            'effective_groups': effective_groups,
            'group_ranges': group_ranges.copy(),
            'grouped_modules': grouped_modules.copy(),
            'multi_string_power': multi_string_power,
            'group_powers': group_powers.copy(),
            'valid_calculation': valid_calculation,
            'traditional_max_power': traditional_max_power,
            'traditional_sum_iv': traditional_sum_iv
        }
        timestamp_data.append(timestamp_info)
        
        # Continue with existing analysis (single-string calculation for comparison)
        combined_voltage = np.zeros_like(currents)
        valid_data_found = False
        sum_iv = 0
        max_power = np.nan
        
        # Create matplotlib figure for visualization
        fig_long, axs_long = plt.subplots(1, 3, figsize=LONG_HOZ_FIGSIZE)
        subplot_titles = ["Recorded MPP values", "Reconstructed I-V curves", "Multi-String I-V curves"]
        subplot_labels = ['(a)', '(b)', '(c)']
        
        for i, (title, label) in enumerate(zip(subplot_titles, subplot_labels)):
            axs_long[i].set_title(title, fontsize=TITLE_SIZE, pad=20)  # Match original font size and padding
            axs_long[i].set_xlabel('Voltage (V)', fontsize=AXIS_LABEL_SIZE)
            axs_long[i].set_ylabel('Current (A)', fontsize=AXIS_LABEL_SIZE)
            axs_long[i].text(0.95, 0.95, label, transform=axs_long[i].transAxes, 
                           fontsize=TEXT_SIZE, ha='right', va='top')
            axs_long[i].tick_params(axis='both', labelsize=AXIS_NUM_SIZE)
        
        axs_long[0].set_xlim(X_LIMIT_MODULE)
        axs_long[0].set_ylim(Y_LIMIT_MODULE)
        axs_long[1].set_xlim(X_LIMIT_MODULE)
        axs_long[1].set_ylim(Y_LIMIT_MODULE)
        vmax_scaled = X_LIMIT_INVERTER[1] 
        axs_long[2].set_xlim((0, vmax_scaled))
        axs_long[2].set_ylim(Y_LIMIT_INVERTER)
        
        # Create color-coded plots for first two subplots
        group_colors, legend_handles, legend_labels = create_color_coded_plots(
            group_assignments, effective_groups, merged_data, idx, 
            reporter_ids, axs_long, currents, module_params, group_ranges
        )
        
        # Plot multi-string I-V curves on the right subplot
        plot_multi_string_iv_curves(
            grouped_modules, merged_data, idx, currents, module_params, axs_long, group_colors
        )
        
        # Add legend for plot (b) in the gap between subplot titles and main title
        if legend_handles and legend_labels:
            fig_long.legend(legend_handles, legend_labels, 
                          bbox_to_anchor=(0.5, 0.7), loc='center', 
                          fontsize=AXIS_NUM_SIZE-3, ncol=len(legend_labels), 
                          framealpha=1, facecolor='white', edgecolor='black', 
                          fancybox=True, shadow=False)
        
        # Calculate traditional series combination for comparison (background calculation)
        for optimiser in reporter_ids:
            optimiser_voltage = merged_data[f'panel_voltage_{optimiser}']
            optimiser_current = merged_data[f'panel_current_{optimiser}']
            panel_temperature = merged_data[f'panel_temperature_{optimiser}']
            
            is_nan_or_zero = (
                optimiser_voltage.iloc[idx] == 0 or
                optimiser_current.iloc[idx] == 0 or
                np.isnan(optimiser_voltage.iloc[idx]) or
                np.isnan(optimiser_current.iloc[idx])
            )
            
            if is_nan_or_zero:
                voltage = np.zeros_like(currents)
                combined_voltage += voltage
                
                # Record NaN parameters
                module_param_row = pd.DataFrame({
                    'Timestamp': [current_timestamp],
                    'Optimizer': [optimiser],
                    'I0': [np.nan], 'Isc': [np.nan], 'Voc': [np.nan],
                    'FF': [np.nan], 'Pmp': [np.nan], 'Imp': [np.nan], 'Vmp': [np.nan]
                })
                if module_param_df.empty:
                    module_param_df = module_param_row
                else:
                    module_param_df = pd.concat([module_param_df, module_param_row], ignore_index=True)
            else:
                valid_data_found = True
                panel_temperature_kelvin = panel_temperature.iloc[idx] + 273.15
                vth = (BOLTZMANN_CONSTANT * panel_temperature_kelvin / ELECTRON_CHARGE 
                       if USE_DYNAMIC_VTH else 0.0259)
                panel_voltage = optimiser_voltage.iloc[idx]
                panel_current = optimiser_current.iloc[idx]
                sum_iv += panel_voltage * panel_current
                
                # Calculate single-diode parameters
                I0_op = I0(panel_current, panel_voltage, module_params['Rs'], module_params['Rsh'], 
                          module_params['n'], module_params['N'], vth)
                IL_op = IL(panel_current, panel_voltage, module_params['Rs'], module_params['Rsh'], 
                          module_params['n'], module_params['N'], vth, I0_op)
                
                params = {
                    'photocurrent': IL_op,
                    'saturation_current': I0_op,
                    'resistance_series': module_params['Rs'],
                    'resistance_shunt': module_params['Rsh'],
                    'nNsVth': module_params['n'] * module_params['N'] * vth
                }
                
                voltage = pvlib.pvsystem.v_from_i(
                    current=currents,
                    photocurrent=params['photocurrent'],
                    saturation_current=params['saturation_current'],
                    resistance_series=params['resistance_series'],
                    resistance_shunt=params['resistance_shunt'],
                    nNsVth=params['nNsVth']
                )
                
                results = pvlib.pvsystem.singlediode(**params)
                isc = results['i_sc']
                voc = results['v_oc']
                pmp = results['p_mp']
                imp = results['i_mp']
                vmp = results['v_mp']
                ff = (pmp / (isc * voc)) if (isc > 0 and voc > 0) else np.nan
                voltage = np.where(currents > isc, 0, voltage)
                combined_voltage += voltage
                
                # Store module parameters
                module_param_row_valid = pd.DataFrame({
                    'Timestamp': [current_timestamp],
                    'Optimizer': [optimiser],
                    'I0': [I0_op], 'Isc': [isc], 'Voc': [voc],
                    'FF': [ff], 'Pmp': [pmp], 'Imp': [imp], 'Vmp': [vmp]
                })
                if module_param_df.empty:
                    module_param_df = module_param_row_valid
                else:
                    module_param_df = pd.concat([module_param_df, module_param_row_valid], ignore_index=True)
        
        # Store traditional series results (background calculation)
        if valid_data_found:
            power = combined_voltage * currents
            max_power_idx = np.argmax(power)
            max_voltage = combined_voltage[max_power_idx]
            max_current = currents[max_power_idx]
            max_power = power[max_power_idx]
            isc_combined = currents[np.where(combined_voltage > 0)[0][-1]] if np.any(combined_voltage > 0) else 0
            voc_combined = combined_voltage[np.where(currents == 0)[0][0]] if len(np.where(currents == 0)[0]) > 0 else 0
            
            # Store results for comparison
            max_power_point = pd.DataFrame({
                'Timestamp': [current_timestamp],
                'Max Voltage (V)': [max_voltage],
                'Max Current (A)': [max_current], 
                'Max Power (W)': [max_power],
                'Voc (V)': [voc_combined],
                'Isc (A)': [isc_combined]
            })
            if max_power_df_combined.empty:
                max_power_df_combined = max_power_point
            else:
                max_power_df_combined = pd.concat([max_power_df_combined, max_power_point], ignore_index=True)
            
            pmppt_row_dup = pd.DataFrame({
                'Timestamp': [current_timestamp], 'Pmppt (W)': [max_power]
            })
            if pmppt_data.empty:
                pmppt_data = pmppt_row_dup
            else:
                pmppt_data = pd.concat([pmppt_data, pmppt_row_dup], ignore_index=True)
        
        # Store sum_iv and multi-string power (note: this appears to be duplicate of earlier storage)
        timestamp_title = current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        iv_sum_row_dup = pd.DataFrame({
            'Timestamp': [timestamp_title], 'Sum of I*V (W)': [sum_iv]
        })
        if iv_sum_data.empty:
            iv_sum_data = iv_sum_row_dup
        else:
            iv_sum_data = pd.concat([iv_sum_data, iv_sum_row_dup], ignore_index=True)
        
        multi_string_row_dup = pd.DataFrame({
            'Timestamp': [timestamp_title], 'Multi_String_Power (W)': [multi_string_power]
        })
        if multi_string_data.empty:
            multi_string_data = multi_string_row_dup
        else:
            multi_string_data = pd.concat([multi_string_data, multi_string_row_dup], ignore_index=True)
        
        # Calculate mismatch losses
        traditional_mismatch = ((sum_iv - max_power) / sum_iv * 100) if sum_iv > 0 else 0
        multi_string_mismatch = ((sum_iv - multi_string_power) / sum_iv * 100) if sum_iv > 0 else 0
        
        # Create plot titles with multi-string information
        title_row1 = f"Site: {site_id} | {timestamp_title} | Groups: {effective_groups}"
        title_row2 = f"Sum MPP: {sum_iv:.1f}W | Trad. Series: {max_power:.1f}W | Multi-String: {multi_string_power:.1f}W"
        title_row3 = f"Trad. Loss: {traditional_mismatch:.2f}% | Multi-String Loss: {multi_string_mismatch:.2f}%"
        
        fig_long.suptitle(f"{title_row1}\n{title_row2}\n{title_row3}", fontsize=TITLE_SIZE, y=0.95)
        
        # Adjust layout with better spacing to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.85])
        file_path = os.path.join(output_dir, f'long_horizontal_{timestamp_title.replace(":", "-").replace(" ", "_")}_grouped.png')
        plt.savefig(file_path, bbox_inches='tight', dpi=150)  # Reduced DPI for faster saving
        plt.close(fig_long)
        image_files.append(file_path)
        
        if idx % 20 == 0:  # Progress update every 20 timestamps
            print(f"  Processed timestamp {idx+1}/{len(merged_data)}: {traditional_mismatch:.2f}% trad, {multi_string_mismatch:.2f}% multi")
    
    print(f"Phase 1 complete: Collected data from {len(timestamp_data)} timestamps")
    
    # ============================================================================
    # PHASE 2: CONSISTENT GROUP DETERMINATION
    # ============================================================================
    print("Phase 2: Determining consistent groups across time series...")
    consistent_groups = determine_consistent_groups(grouping_history, reporter_ids)
    
    print(f"Consistent group assignments determined:")
    for reporter_id, group_id in consistent_groups.items():
        print(f"  Reporter {reporter_id} -> Group {group_id}")
    
    # ============================================================================
    # PHASE 3: CONSISTENT VISUALIZATION AND FINAL CALCULATIONS
    # ============================================================================
    print("Phase 3: Creating visualizations with consistent grouping...")
    
    # Create grouped modules based on consistent grouping
    consistent_grouped_modules = {}
    for reporter_id, group_id in consistent_groups.items():
        if group_id not in consistent_grouped_modules:
            consistent_grouped_modules[group_id] = []
        consistent_grouped_modules[group_id].append(reporter_id)
    
    # Process visualizations using consistent grouping
    image_files = []  # Reset image files for consistent grouping plots
    for timestamp_info in timestamp_data:
        idx = timestamp_info['idx']
        current_timestamp = timestamp_info['timestamp']
        
        # Use consistent grouping for visualization
        group_assignments = consistent_groups  # Use consistent groups instead of timestamp-specific
        effective_groups = len(consistent_grouped_modules)
        grouped_modules = consistent_grouped_modules
        
        # Create visualization plots with consistent grouping
        fig_long, axs_long = plt.subplots(1, 3, figsize=LONG_HOZ_FIGSIZE)
        
        # Configure subplots with better spacing
        subplot_titles = ["Recorded MPP values", "Reconstructed I-V curves", "Multi-String I-V curves"]
        subplot_labels = ['(a)', '(b)', '(c)']
        
        for i, (title, label) in enumerate(zip(subplot_titles, subplot_labels)):
            axs_long[i].set_title(title, fontsize=TITLE_SIZE, pad=20)
            axs_long[i].set_xlabel('Voltage (V)', fontsize=AXIS_LABEL_SIZE)
            axs_long[i].set_ylabel('Current (A)', fontsize=AXIS_LABEL_SIZE)
            axs_long[i].text(0.95, 0.95, label, transform=axs_long[i].transAxes, 
                           fontsize=TEXT_SIZE, ha='right', va='top')
            axs_long[i].tick_params(axis='both', labelsize=AXIS_NUM_SIZE)

        axs_long[0].set_xlim(X_LIMIT_MODULE)
        axs_long[0].set_ylim(Y_LIMIT_MODULE)
        axs_long[1].set_xlim(X_LIMIT_MODULE)
        axs_long[1].set_ylim(Y_LIMIT_MODULE)
        vmax_scaled = X_LIMIT_INVERTER[1] 
        axs_long[2].set_xlim((0, vmax_scaled))
        axs_long[2].set_ylim(Y_LIMIT_INVERTER)
        
        # Create color-coded plots for first two subplots using consistent grouping
        # Note: group_ranges needs to be recalculated for consistent groups
        consistent_group_ranges = []
        for group_id in range(1, effective_groups + 1):
            reporter_list = consistent_grouped_modules[group_id]
            group_currents = []
            for reporter_id in reporter_list:
                current_col = f'panel_current_{reporter_id}'
                if current_col in merged_data.columns:
                    current_val = merged_data[current_col].iloc[idx]
                    if not (np.isnan(current_val) or current_val <= 0):
                        group_currents.append(current_val)
            if group_currents:
                consistent_group_ranges.append((min(group_currents), max(group_currents)))
            else:
                consistent_group_ranges.append((0, 0))
        
        group_colors, legend_handles, legend_labels = create_color_coded_plots(
            group_assignments, effective_groups, merged_data, idx, 
            reporter_ids, axs_long, currents, module_params, consistent_group_ranges
        )
        
        # Plot multi-string I-V curves using consistent grouping
        plot_multi_string_iv_curves(
            grouped_modules, merged_data, idx, currents, module_params, axs_long, group_colors
        )
        
        # Add legend for plot (b)
        if legend_handles and legend_labels:
            fig_long.legend(legend_handles, legend_labels, 
                          bbox_to_anchor=(0.5, 0.7), loc='center', 
                          fontsize=AXIS_NUM_SIZE-3, ncol=len(legend_labels), 
                          framealpha=1, facecolor='white', edgecolor='black', 
                          fancybox=True, shadow=False)
        
        # Calculate consistent multi-string power
        consistent_multi_string_power, consistent_group_powers, _ = calculate_multi_string_series_power(
            consistent_grouped_modules, merged_data, idx, currents, module_params
        )
        
        # Calculate traditional mismatch for comparison
        sum_iv = sum(merged_data[f'panel_voltage_{opt}'].iloc[idx] * merged_data[f'panel_current_{opt}'].iloc[idx] 
                     for opt in reporter_ids 
                     if f'panel_voltage_{opt}' in merged_data.columns and f'panel_current_{opt}' in merged_data.columns
                     and not (np.isnan(merged_data[f'panel_voltage_{opt}'].iloc[idx]) or np.isnan(merged_data[f'panel_current_{opt}'].iloc[idx])))
        
        # Use original timestamp-based calculations for traditional comparison
        original_max_power = timestamp_info.get('traditional_max_power', np.nan)  # Will need to store this in Phase 1
        traditional_mismatch = ((sum_iv - original_max_power) / sum_iv * 100) if sum_iv > 0 and not np.isnan(original_max_power) else 0
        consistent_multi_string_mismatch = ((sum_iv - consistent_multi_string_power) / sum_iv * 100) if sum_iv > 0 else 0
        
        # Create plot titles with consistent multi-string information
        timestamp_title = current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        title_row1 = f"Site: {site_id} | {timestamp_title} | Consistent Groups: {effective_groups}"
        title_row2 = f"Sum MPP: {sum_iv:.1f}W | Trad. Series: {original_max_power:.1f}W | Consistent Multi-String: {consistent_multi_string_power:.1f}W"
        title_row3 = f"Trad. Loss: {traditional_mismatch:.2f}% | Consistent Multi-String Loss: {consistent_multi_string_mismatch:.2f}%"
        
        fig_long.suptitle(f"{title_row1}\n{title_row2}\n{title_row3}", fontsize=TITLE_SIZE, y=0.95)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.85])
        file_path = os.path.join(output_dir, f'long_horizontal_{timestamp_title.replace(":", "-").replace(" ", "_")}_consistent_grouped.png')
        plt.savefig(file_path, bbox_inches='tight', dpi=150)
        plt.close(fig_long)
        image_files.append(file_path)
        
        # Update multi_string_data with consistent power values
        if idx < len(multi_string_data):
            # Update existing row with proper column reference
            multi_string_data.loc[multi_string_data.index[idx], 'Multi_String_Power (W)'] = consistent_multi_string_power
        else:
            # Append new row with proper structure
            new_row = pd.DataFrame({
                'Timestamp': [timestamp_title], 
                'Multi_String_Power (W)': [consistent_multi_string_power]
            })
            multi_string_data = pd.concat([multi_string_data, new_row], ignore_index=True)
        
        if idx % 20 == 0:
            print(f"  Processed consistent visualization {idx+1}/{len(timestamp_data)}: {consistent_multi_string_mismatch:.2f}% consistent multi")
    
    print(f"Phase 3 complete: Created {len(image_files)} consistent visualizations")
    
    # Export results
    export_enhanced_results(
        iv_sum_data, pmppt_data, multi_string_data, max_power_df_combined,
        module_param_df, grouping_data, image_files, output_dir, site_id, season
    )


def export_enhanced_results(iv_sum_data: pd.DataFrame,
                          pmppt_data: pd.DataFrame, 
                          multi_string_data: pd.DataFrame,
                          max_power_df_combined: pd.DataFrame,
                          module_param_df: pd.DataFrame,
                          grouping_data: pd.DataFrame,
                          image_files: List[str],
                          output_dir: str,
                          site_id: str,
                          season: str) -> None:
    """Export all analysis results with enhanced multi-string metrics."""
    
    print(f"Exporting results to {output_dir}...")
    
    # Create GIF from plots
    if image_files:
        gif_path = os.path.join(output_dir, 'combined_iv_curves_grouped.gif')
        try:
            with imageio.get_writer(gif_path, mode='I', duration=200, loop=0) as writer:
                for filename in image_files:
                    image = imageio.imread(filename)
                    writer.append_data(image)
            print(f"GIF saved: {gif_path}")
        except Exception as e:
            print(f"Warning: Could not create GIF: {str(e)}")
    
    # Combine all power data
    try:
        combined_data = pd.merge(iv_sum_data, pmppt_data[['Timestamp', 'Pmppt (W)']], on='Timestamp', how='outer')
        combined_data = pd.merge(combined_data, multi_string_data[['Timestamp', 'Multi_String_Power (W)']], on='Timestamp', how='outer')
        
        # Add metadata
        combined_data['Season'] = season
        combined_data['Site ID'] = site_id
        
        # Calculate mismatch metrics
        combined_data['Traditional_Mismatch_%'] = (
            (combined_data['Sum of I*V (W)'] - combined_data['Pmppt (W)']) / 
            combined_data['Sum of I*V (W)'] * 100
        ).fillna(0)
        
        combined_data['Consistent_Multi_String_Mismatch_%'] = (
            (combined_data['Sum of I*V (W)'] - combined_data['Multi_String_Power (W)']) / 
            combined_data['Sum of I*V (W)'] * 100
        ).fillna(0)
        
        combined_data['Improvement_%'] = (
            combined_data['Traditional_Mismatch_%'] - combined_data['Consistent_Multi_String_Mismatch_%']
        )
        
        # Export enhanced combined data
        excel_file = os.path.join(output_dir, f'combined_data_enhanced_{season}_{site_id}.xlsx')
        combined_data.to_excel(excel_file, index=False)
        print(f"Enhanced combined data: {excel_file}")
        
        # Generate power comparison plot
        try:
            create_power_comparison_plot(iv_sum_data, pmppt_data, multi_string_data, site_id, season, output_dir)
        except Exception as e:
            print(f"Warning: Could not create power comparison plot: {str(e)}")
        
        # Generate simplified power comparison plot (Sum of MPP vs Multi-string only)
        try:
            create_simplified_power_comparison_plot(iv_sum_data, multi_string_data, site_id, season, output_dir)
        except Exception as e:
            print(f"Warning: Could not create simplified power comparison plot: {str(e)}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        total_sum_iv = combined_data['Sum of I*V (W)'].sum()
        total_pmppt = combined_data['Pmppt (W)'].sum()
        total_multi = combined_data['Multi_String_Power (W)'].sum()
        
        traditional_loss = (total_sum_iv - total_pmppt) / total_sum_iv * 100 if total_sum_iv > 0 else 0
        multi_string_loss = (total_sum_iv - total_multi) / total_sum_iv * 100 if total_sum_iv > 0 else 0
        improvement = traditional_loss - multi_string_loss
        
        print(f"Site ID: {site_id}")
        print(f"Season: {season}")
        print(f"Total Sum of MPP Power: {total_sum_iv:.2f} W")
        print(f"Total Traditional Series Power: {total_pmppt:.2f} W")
        print(f"Total Consistent Multi-String Power: {total_multi:.2f} W")
        print(f"Traditional Mismatch Loss: {traditional_loss:.2f}%")
        print(f"Consistent Multi-String Mismatch Loss: {multi_string_loss:.2f}%")
        print(f"Improvement: {improvement:.2f} percentage points")
        print("="*60)
        
    except Exception as e:
        print(f"Warning: Could not create enhanced combined data: {str(e)}")
    
    # Export individual datasets
    try:
        iv_sum_data.to_excel(os.path.join(output_dir, 'iv_sum_data.xlsx'), index=False)
        pmppt_data.to_excel(os.path.join(output_dir, 'pmppt_data.xlsx'), index=False)
        multi_string_data.to_excel(os.path.join(output_dir, 'multi_string_data.xlsx'), index=False)
        module_param_df.to_csv(os.path.join(output_dir, 'module_param_df.csv'), index=False)
        
        # Export detailed grouping analysis
        grouping_file = os.path.join(output_dir, f'multi_string_analysis_{site_id}.csv')
        grouping_data.to_csv(grouping_file, index=False)
        print(f"Multi-string analysis: {grouping_file}")
        
        print("All individual datasets exported successfully")
        
    except Exception as e:
        print(f"Warning: Could not export some individual datasets: {str(e)}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("Multi-Orientation Solar Array Mismatch Analysis")
    print("=" * 60)
    
    # Process all multi-orientation sites
    run_multi_orientation_analysis(
        site_ids=None,  # Process all multi-orientation sites
        seasons=['spring'],  # Focus on spring season
        num_days_to_plot=10  # Limit to 5 days for faster processing
    )
    
    # Alternative: Process specific sites
    # run_multi_orientation_analysis(
    #     site_ids=['3455043', '4111492'],  # Specific sites only
    #     seasons=['spring', 'summer'],
    #     num_days_to_plot=10
    # )