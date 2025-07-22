#!/usr/bin/env python3
"""
Daily Mismatch Loss Analysis Script

This script analyzes solar energy mismatch loss on a daily basis, providing more 
granular insights than seasonal analysis. It processes raw data, calculates daily 
metrics, merges with site metadata, and generates visualizations.

Author: Generated for Solar Edge Research Project
Date: 2025-01-22
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from geopy.geocoders import Nominatim
from kgcpy import lookupCZ

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION VARIABLES
# ==============================================================================

# Input data paths
RESULTS_FOLDER = r"C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\24_09_24_Solar_Edge\Results\v_from_i_combined\25_06_24_Results"
SITE_SUMMARY_FILE = r"C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\24_09_24_Solar_Edge\Data\25_05_01_Newsites_summary.xlsx"
OLD_RESULTS_FILE = r"C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\24_09_24_Solar_Edge\Results\25_07_02_Old_Sites\25_07_01_old_summary.xlsx"

# Output directory
OUTPUT_FOLDER = r"C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\24_09_24_Solar_Edge\Results\daily_analysis_results"

# Plot styling parameters
FIGURE_SIZE = (10, 6)
DPI = 300
FONT_SIZE = 12
TITLE_SIZE = 14

# Köppen–Geiger climate zone mapping
KG_FULL_NAMES = {
    "Af": "Tropical rainforest",
    "Am": "Tropical monsoon", 
    "Aw": "Tropical savanna",
    "As": "Tropical dry savanna",
    "BWk": "Cold desert",
    "BWh": "Hot desert",
    "BSk": "Cold semi-arid",
    "BSh": "Hot semi-arid",
    "Csa": "Hot-summer Mediterranean",
    "Csb": "Warm-summer Mediterranean",
    "Csc": "Cold-summer Mediterranean",
    "Cwa": "Monsoon-influenced humid subtropical",
    "Cwb": "Subtropical highland oceanic",
    "Cwc": "Cold subtropical highland",
    "Cfa": "Humid subtropical",
    "Cfb": "Oceanic",
    "Cfc": "Subpolar oceanic",
    "Dsa": "Hot-summer humid continental",
    "Dsb": "Warm-summer humid continental",
    "Dsc": "Subarctic",
    "Dsd": "Extremely cold subarctic",
    "Dwa": "Monsoon-influenced hot-summer humid continental",
    "Dwb": "Monsoon-influenced warm-summer humid continental",
    "Dwc": "Monsoon-influenced subarctic",
    "Dwd": "Monsoon-influenced extremely cold subarctic",
    "Dfa": "Hot-summer humid continental",
    "Dfb": "Warm-summer humid continental", 
    "Dfc": "Subarctic",
    "Dfd": "Extremely cold subarctic",
    "ET": "Tundra",
    "EF": "Ice cap"
}

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def create_output_directory():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output directory: {OUTPUT_FOLDER}")
    else:
        print(f"Output directory already exists: {OUTPUT_FOLDER}")

def find_data_files(results_folder):
    """
    Find all relevant data files in the results folder.
    
    Args:
        results_folder (str): Path to the results folder
        
    Returns:
        list: List of tuples containing (site_id, season, file_path)
    """
    data_files = []
    
    if not os.path.exists(results_folder):
        print(f"Warning: Results folder does not exist: {results_folder}")
        return data_files
    
    for folder_name in os.listdir(results_folder):
        folder_path = os.path.join(results_folder, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
            
        # Extract site_id and season from folder name
        try:
            parts = folder_name.split('_')
            site_id = parts[0]
            season = parts[1]
        except (IndexError, ValueError):
            print(f"Warning: Could not parse folder name: {folder_name}")
            continue
        
        # Find files containing both 'combined_data' and 'no_diode'
        for file_name in os.listdir(folder_path):
            if ('combined_data' in file_name and 'no_diode' in file_name and 
                (file_name.endswith('.csv') or file_name.endswith('.xlsx'))):
                file_path = os.path.join(folder_path, file_name)
                data_files.append((site_id, season, file_path))
                break
    
    print(f"Found {len(data_files)} data files to process")
    return data_files

def calculate_fft_period(timestamps, power_data):
    """
    Calculate centroid frequency and mean period using FFT analysis.
    
    Args:
        timestamps (pd.Series): Timestamp data
        power_data (pd.Series): Power data for FFT analysis
        
    Returns:
        float: Mean period in hours
    """
    try:
        # Calculate time step
        dt = (timestamps.iloc[1] - timestamps.iloc[0]).total_seconds()
        
        # Prepare data for FFT
        y = power_data.values
        N = len(y)
        
        if N < 2:
            return np.nan
        
        # Remove mean and perform FFT
        y_centered = y - np.mean(y)
        y_fft = np.fft.fft(y_centered)
        freqs = np.fft.fftfreq(N, d=dt)
        
        # Only use positive frequencies
        mask = freqs > 0
        freqs_p = freqs[mask]
        amp_p = np.abs(y_fft[mask])
        power = amp_p**2
        
        # Calculate centroid frequency
        if np.sum(power) > 0:
            f_centroid = np.sum(freqs_p * power) / np.sum(power)
            if f_centroid > 0:
                T_centroid_s = 1.0 / f_centroid
                T_centroid_h = T_centroid_s / 3600.0 * 2  # Convert to hours
                return T_centroid_h
        
        return np.nan
        
    except Exception as e:
        print(f"Warning: FFT calculation failed: {e}")
        return np.nan

def calculate_mismatch_loss(sum_iv, pmppt):
    """
    Calculate mismatch loss percentage.
    
    Args:
        sum_iv (float): Sum of I*V power
        pmppt (float): Series connection power
        
    Returns:
        float: Mismatch loss percentage
    """
    if sum_iv == 0 or np.isnan(sum_iv) or np.isnan(pmppt):
        return np.nan
    
    mismatch_loss = (sum_iv - pmppt) / sum_iv * 100
    return max(0, min(100, mismatch_loss))  # Clamp between 0 and 100

def process_daily_data(file_path, site_id, season):
    """
    Process a single data file to extract daily metrics.
    
    Args:
        file_path (str): Path to the data file
        site_id (str): Site identifier
        season (str): Season identifier
        
    Returns:
        pd.DataFrame: Daily results for this site/season
    """
    daily_results = []
    
    try:
        # Load data file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Group by date
        df['Date'] = df['Timestamp'].dt.date
        
        for date, group in df.groupby('Date'):
            if len(group) < 2:  # Need at least 2 points for FFT
                continue
                
            # Calculate mean period using FFT
            mean_period = calculate_fft_period(group['Timestamp'], group['Sum of I*V (W)'])
            
            # Calculate daily mismatch loss
            sum_iv_total = group['Sum of I*V (W)'].sum()
            pmppt_total = group['Pmppt (W)'].sum()
            mismatch_loss = calculate_mismatch_loss(sum_iv_total, pmppt_total)
            
            # Only add valid results
            if not np.isnan(mismatch_loss):
                daily_results.append({
                    'Site ID': site_id,
                    'Season': season,
                    'Date': date,
                    'Mean Period (h)': mean_period,
                    'Mismatch Loss (%)': mismatch_loss
                })
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    
    return pd.DataFrame(daily_results)

def add_climate_zones(df):
    """
    Add climate zone information to the dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame with site information
        
    Returns:
        pd.DataFrame: DataFrame with climate zone information added
    """
    geolocator = Nominatim(user_agent="daily-mismatch-analysis")
    
    if 'Climate Zone' not in df.columns:
        df['Climate Zone'] = None
    
    for idx, row in df.iterrows():
        if pd.isna(row.get('Climate Zone')) and pd.notna(row.get('Address')):
            try:
                location = geolocator.geocode(row['Address'], timeout=10)
                if location:
                    kg_code = lookupCZ(location.latitude, location.longitude)
                    kg_full = KG_FULL_NAMES.get(kg_code, kg_code)
                    df.at[idx, 'Climate Zone'] = kg_full
                else:
                    df.at[idx, 'Climate Zone'] = 'Not found'
            except Exception as e:
                print(f"Error geocoding {row['Address']}: {e}")
                df.at[idx, 'Climate Zone'] = 'Error'
    
    return df

# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def create_mismatch_distribution_plot(df, output_folder):
    """Create histogram of mismatch loss distribution."""
    plt.figure(figsize=FIGURE_SIZE)
    
    mismatch_data = df['Mismatch Loss (%)'].dropna()
    mean_val = mismatch_data.mean()
    std_val = mismatch_data.std()
    
    plt.hist(mismatch_data, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Mismatch Loss (%)', fontsize=FONT_SIZE)
    plt.ylabel('Frequency', fontsize=FONT_SIZE)
    plt.title(f'Daily Mismatch Loss Distribution\nMean: {mean_val:.2f}%, Std: {std_val:.2f}%', 
              fontsize=TITLE_SIZE)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'mismatch_loss_distribution.png'), 
                dpi=DPI, bbox_inches='tight')
    plt.close()

def create_mismatch_vs_period_plots(df, output_folder):
    """Create scatter plots of mismatch loss vs mean period."""
    # Plot by Orientation
    plt.figure(figsize=FIGURE_SIZE)
    
    for orientation in df['Orientation'].unique():
        if pd.notna(orientation):
            subset = df[df['Orientation'] == orientation]
            plt.scatter(subset['Mean Period (h)'], subset['Mismatch Loss (%)'], 
                       label=orientation, alpha=0.6)
            
            # Add linear regression fit
            valid_data = subset[['Mean Period (h)', 'Mismatch Loss (%)']].dropna()
            if len(valid_data) > 1:
                z = np.polyfit(valid_data['Mean Period (h)'], valid_data['Mismatch Loss (%)'], 1)
                p = np.poly1d(z)
                x_range = np.linspace(valid_data['Mean Period (h)'].min(), 
                                    valid_data['Mean Period (h)'].max(), 100)
                plt.plot(x_range, p(x_range), '--', alpha=0.8, 
                        label=f'{orientation} fit (slope={z[0]:.2f})')
    
    plt.xlabel('Mean Period (h)', fontsize=FONT_SIZE)
    plt.ylabel('Mismatch Loss (%)', fontsize=FONT_SIZE)
    plt.title('Mismatch Loss vs Mean Period (by Orientation)', fontsize=TITLE_SIZE)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'mismatch_vs_period_orientation.png'), 
                dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # Plot by Shading
    plt.figure(figsize=FIGURE_SIZE)
    
    for shade in df['Shade'].unique():
        if pd.notna(shade):
            subset = df[df['Shade'] == shade]
            plt.scatter(subset['Mean Period (h)'], subset['Mismatch Loss (%)'], 
                       label=f'Shade: {shade}', alpha=0.6)
            
            # Add linear regression fit
            valid_data = subset[['Mean Period (h)', 'Mismatch Loss (%)']].dropna()
            if len(valid_data) > 1:
                z = np.polyfit(valid_data['Mean Period (h)'], valid_data['Mismatch Loss (%)'], 1)
                p = np.poly1d(z)
                x_range = np.linspace(valid_data['Mean Period (h)'].min(), 
                                    valid_data['Mean Period (h)'].max(), 100)
                plt.plot(x_range, p(x_range), '--', alpha=0.8, 
                        label=f'{shade} fit (slope={z[0]:.2f})')
    
    plt.xlabel('Mean Period (h)', fontsize=FONT_SIZE)
    plt.ylabel('Mismatch Loss (%)', fontsize=FONT_SIZE)
    plt.title('Mismatch Loss vs Mean Period (by Shading)', fontsize=TITLE_SIZE)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'mismatch_vs_period_shading.png'), 
                dpi=DPI, bbox_inches='tight')
    plt.close()

def create_grouped_bar_charts(df, output_folder):
    """Create grouped bar charts for different categories."""
    
    # Climate Zone and Shading
    if 'Climate Zone' in df.columns and 'Shade' in df.columns:
        plt.figure(figsize=(12, 6))
        
        climate_shade_stats = df.groupby(['Climate Zone', 'Shade'])['Mismatch Loss (%)'].agg(['mean', 'std']).reset_index()
        
        if not climate_shade_stats.empty:
            climate_zones = climate_shade_stats['Climate Zone'].unique()
            x = np.arange(len(climate_zones))
            width = 0.35
            
            shade_yes = climate_shade_stats[climate_shade_stats['Shade'] == 'Yes']
            shade_no = climate_shade_stats[climate_shade_stats['Shade'] == 'No']
            
            plt.bar(x - width/2, shade_yes['mean'], width, yerr=shade_yes['std'], 
                   label='Shaded', capsize=5, alpha=0.8)
            plt.bar(x + width/2, shade_no['mean'], width, yerr=shade_no['std'], 
                   label='Unshaded', capsize=5, alpha=0.8)
            
            plt.xlabel('Climate Zone', fontsize=FONT_SIZE)
            plt.ylabel('Mean Mismatch Loss (%)', fontsize=FONT_SIZE)
            plt.title('Mean Mismatch Loss by Climate Zone and Shading', fontsize=TITLE_SIZE)
            plt.xticks(x, climate_zones, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'climate_zone_shading_chart.png'), 
                       dpi=DPI, bbox_inches='tight')
            plt.close()
    
    # Seasonal Chart
    plt.figure(figsize=FIGURE_SIZE)
    seasonal_stats = df.groupby('Season')['Mismatch Loss (%)'].agg(['mean', 'std']).reset_index()
    
    plt.bar(seasonal_stats['Season'], seasonal_stats['mean'], 
           yerr=seasonal_stats['std'], capsize=5, alpha=0.8)
    plt.xlabel('Season', fontsize=FONT_SIZE)
    plt.ylabel('Mean Mismatch Loss (%)', fontsize=FONT_SIZE)
    plt.title('Mean Mismatch Loss by Season', fontsize=TITLE_SIZE)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'seasonal_chart.png'), 
               dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # Orientation Chart
    if 'Orientation' in df.columns:
        plt.figure(figsize=FIGURE_SIZE)
        orientation_stats = df.groupby('Orientation')['Mismatch Loss (%)'].agg(['mean', 'std']).reset_index()
        
        plt.bar(orientation_stats['Orientation'], orientation_stats['mean'], 
               yerr=orientation_stats['std'], capsize=5, alpha=0.8)
        plt.xlabel('Orientation', fontsize=FONT_SIZE)
        plt.ylabel('Mean Mismatch Loss (%)', fontsize=FONT_SIZE)
        plt.title('Mean Mismatch Loss by Orientation', fontsize=TITLE_SIZE)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'orientation_chart.png'), 
                   dpi=DPI, bbox_inches='tight')
        plt.close()

# ==============================================================================
# MAIN EXECUTION FUNCTION
# ==============================================================================

def main():
    """Main execution function."""
    print("Starting Daily Mismatch Loss Analysis...")
    print("=" * 50)
    
    # Create output directory
    create_output_directory()
    
    # Initialize results DataFrame
    daily_results_df = pd.DataFrame(columns=['Site ID', 'Season', 'Date', 'Mean Period (h)', 'Mismatch Loss (%)'])
    
    # Find all data files
    data_files = find_data_files(RESULTS_FOLDER)
    
    if not data_files:
        print("No data files found. Exiting.")
        return
    
    # Process each data file
    print("\nProcessing data files...")
    for i, (site_id, season, file_path) in enumerate(data_files):
        print(f"Processing {i+1}/{len(data_files)}: Site {site_id}, Season {season}")
        
        daily_data = process_daily_data(file_path, site_id, season)
        if not daily_data.empty:
            daily_results_df = pd.concat([daily_results_df, daily_data], ignore_index=True)
    
    if daily_results_df.empty:
        print("No valid daily data found. Exiting.")
        return
    
    print(f"\nProcessed {len(daily_results_df)} daily records from {len(data_files)} sites/seasons")
    
    # Load site summary data and merge
    print("\nMerging with site metadata...")
    try:
        site_summary_df = pd.read_excel(SITE_SUMMARY_FILE)
        site_summary_df['Site ID'] = site_summary_df['Site ID'].astype(str)
        daily_results_df['Site ID'] = daily_results_df['Site ID'].astype(str)
        
        # Merge with site summary
        enriched_df = pd.merge(daily_results_df, site_summary_df, on='Site ID', how='left')
        
        # Add climate zone information
        print("Adding climate zone information...")
        enriched_df = add_climate_zones(enriched_df)
        
    except Exception as e:
        print(f"Warning: Could not merge with site metadata: {e}")
        enriched_df = daily_results_df
    
    # Export to Excel
    excel_path = os.path.join(OUTPUT_FOLDER, 'daily_mismatch_summary.xlsx')
    enriched_df.to_excel(excel_path, index=False)
    print(f"\nExported results to: {excel_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    create_mismatch_distribution_plot(enriched_df, OUTPUT_FOLDER)
    print("Created mismatch loss distribution plot")
    
    create_mismatch_vs_period_plots(enriched_df, OUTPUT_FOLDER)
    print("Created mismatch vs period scatter plots")
    
    create_grouped_bar_charts(enriched_df, OUTPUT_FOLDER)
    print("Created grouped bar charts")
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total daily records: {len(enriched_df)}")
    print(f"Unique sites: {enriched_df['Site ID'].nunique()}")
    print(f"Date range: {enriched_df['Date'].min()} to {enriched_df['Date'].max()}")
    print(f"Mean mismatch loss: {enriched_df['Mismatch Loss (%)'].mean():.2f}%")
    print(f"Std mismatch loss: {enriched_df['Mismatch Loss (%)'].std():.2f}%")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()