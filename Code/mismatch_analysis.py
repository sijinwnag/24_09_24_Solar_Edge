"""
Mismatch Analysis Module for Solar PV Systems

This module provides a comprehensive class for analyzing mismatch losses in SolarEdge systems
by reconstructing I-V curves from MPP telemetry data using single-diode models.

Core Research Question: How do module-level mismatches and bypass diode activation 
affect overall system power output in real-world SolarEdge installations?

Author: Refactored from 25_04_27_Mismatch_results_generator.ipynb
Date: 2025-01-04
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
from typing import Dict, List, Tuple, Optional
import logging


class MismatchAnalysis:
    """
    A comprehensive class for analyzing mismatch losses in solar PV systems.
    
    This class encapsulates the complete workflow for:
    - Loading and preprocessing optimizer telemetry data
    - Extracting module parameters from .PAN files
    - Reconstructing I-V curves using single-diode models
    - Calculating mismatch losses (sum-of-MPP vs series-connection)
    - Generating visualizations and results
    """
    
    def __init__(self, site_id: str, season: str, data_dir: str, results_dir: str, summary_dir: str):
        """
        Initialize the MismatchAnalysis object.
        
        Args:
            site_id (str): The ID of the site to be analyzed
            season (str): The season for the analysis (e.g., 'spring', 'summer')
            data_dir (str): The root directory containing the raw data
            results_dir (str): The root directory where results will be saved
            summary_dir (str): The path to the site summary Excel file
        """
        # Core attributes
        self.site_id = site_id
        self.season = season
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.summary_dir = summary_dir
        
        # Load site summary data
        self.summary_df = pd.read_excel(summary_dir, sheet_name='Sheet1')
        
        # Initialize data containers
        self.merged_data = pd.DataFrame()
        self.reporter_ids = []
        self.module_params = {}
        
        # Initialize result containers
        self.max_power_df_combined = pd.DataFrame(columns=[
            'Timestamp', 'Max Voltage (V)', 'Max Current (A)', 'Max Power (W)', 'Voc (V)', 'Isc (A)'
        ])
        self.pmppt_data = pd.DataFrame(columns=['Timestamp', 'Pmppt (W)'])
        self.module_param_df = pd.DataFrame(columns=[
            'Timestamp', 'Optimizer', 'I0', 'Isc', 'Voc', 'FF', 'Pmp', 'Imp', 'Vmp'
        ])
        self.iv_sum_data = pd.DataFrame(columns=['Timestamp', 'Sum of I*V (W)'])
        
        # Analysis parameters
        self.currents = np.linspace(0, 17, 100)  # Current range for IV curve reconstruction
        self.num_days_to_plot = 10  # Default number of days to analyze
        
        # Physics constants
        self.boltzmann_constant = const.Boltzmann
        self.electron_charge = const.e
        
        # Configuration options
        self.use_dynamic_vth = True  # Calculate thermal voltage from actual temperature
        self.use_a_T = True  # Use ambient temperature when module temperature sensors appear faulty
        
        # Plotting parameters
        self.axis_label_size = 20
        self.axis_num_size = 20
        self.text_size = 20
        self.title_size = 22
        self.figure_size = (6, 6)
        self.long_hoz_figsize = (12, 6)
        
        # Plot limits
        self.y_limit_module = (0, 15)
        self.x_limit_module = (0, 60)
        self.y_limit_inverter = (0, 17)
        self.x_limit_inverter = (0, 1200)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized MismatchAnalysis for site {site_id}, season {season}")

    def load_and_prepare_data(self):
        """
        Load, parse, clean, and merge the raw optimizer data for the specified site and season.
        
        This method handles:
        - Site directory identification
        - Hemisphere detection and seasonal mapping
        - CSV file loading (single or multiple files)
        - Timestamp synchronization to 5-minute intervals
        - Data merging across all optimizers
        """
        self.logger.info("Starting data loading and preparation...")
        
        # Find the site directory
        site_folders = [d for d in os.listdir(self.data_dir) if self.site_id in d]
        if not site_folders:
            raise ValueError(f"No directory found for site_id: {self.site_id}")
        
        site_dir = os.path.join(self.data_dir, site_folders[0])
        self.logger.info(f"Found site directory: {site_dir}")
        
        # Determine hemisphere and seasonal mapping
        site_info = self.summary_df[self.summary_df['Site ID'] == int(self.site_id)]
        if site_info.empty:
            raise ValueError(f"Site ID {self.site_id} not found in summary file")
        
        country = site_info['Country'].values[0]
        season_months = self._get_seasonal_mapping(country)
        
        # Find season directory
        season_dir = self._find_season_directory(site_dir, season_months)
        self.logger.info(f"Season directory found: {season_dir}")
        
        # Load CSV files
        dataframes, reporter_ids = self._load_csv_files(season_dir)
        self.reporter_ids = reporter_ids
        
        # Synchronize timestamps and merge data
        self.merged_data = self._synchronize_and_merge_data(dataframes)
        
        # Handle temperature sensor substitution if needed
        if self.use_a_T:
            self._substitute_temperature_data()
        
        self.logger.info(f"Data loading completed. Found {len(self.reporter_ids)} optimizers")
        self.logger.info(f"Merged data shape: {self.merged_data.shape}")

    def _get_seasonal_mapping(self, country: str) -> Dict[str, List[str]]:
        """Get season-to-months mapping based on hemisphere."""
        if country == 'Australia':
            return {
                'summer': ['december', 'january', 'february'],
                'autumn': ['march', 'april', 'may'],
                'winter': ['june', 'july', 'august'],
                'spring': ['september', 'october', 'november']
            }
        else:
            return {
                'summer': ['june', 'july', 'august'],
                'autumn': ['september', 'october', 'november'],
                'winter': ['december', 'january', 'february'],
                'spring': ['march', 'april', 'may']
            }

    def _find_season_directory(self, site_dir: str, season_months: Dict[str, List[str]]) -> str:
        """Find the directory containing seasonal data."""
        season_lower = self.season.lower()
        season_dir_candidates = [
            d for d in os.listdir(site_dir)
            if (season_lower in d.lower() or 
                any(month in d.lower() for month in season_months.get(season_lower, [])))
        ]
        
        if not season_dir_candidates:
            raise ValueError(f"No folder found for season '{self.season}' in site directory")
        
        return os.path.join(site_dir, season_dir_candidates[0])

    def _load_csv_files(self, season_dir: str) -> Tuple[List[pd.DataFrame], List[str]]:
        """Load and process CSV files containing optimizer data."""
        csv_files = [f for f in os.listdir(season_dir) if 'optimizer_data' in f and f.endswith('.csv')]
        
        if not csv_files:
            raise ValueError("No optimizer data CSV files found in season directory")
        
        dataframes = []
        reporter_ids = []
        
        timestamp_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%m/%d/%Y %H:%M",
            "%Y-%d-%m %H:%M:%S",
            None  # Let pandas infer
        ]
        
        if len(csv_files) == 1:
            # Single CSV file - check for reporter_id column
            dataframes, reporter_ids = self._process_single_csv(csv_files[0], season_dir, timestamp_formats)
        else:
            # Multiple CSV files - each represents one optimizer
            dataframes, reporter_ids = self._process_multiple_csvs(csv_files, season_dir, timestamp_formats)
        
        return dataframes, reporter_ids

    def _process_single_csv(self, file: str, season_dir: str, timestamp_formats: List) -> Tuple[List[pd.DataFrame], List[str]]:
        """Process a single CSV file that may contain multiple optimizers."""
        file_path = os.path.join(season_dir, file)
        df = pd.read_csv(file_path)
        dataframes = []
        reporter_ids = []
        
        if 'reporter_id' in df.columns:
            # Split by reporter_id
            unique_reporters = df['reporter_id'].unique()
            for reporter in unique_reporters:
                df_rep = df[df['reporter_id'] == reporter].copy()
                df_processed = self._process_dataframe(df_rep, str(reporter), timestamp_formats)
                dataframes.append(df_processed)
                reporter_ids.append(str(reporter))
        else:
            # Single optimizer file
            default_reporter = "default"
            df_processed = self._process_dataframe(df, default_reporter, timestamp_formats)
            dataframes.append(df_processed)
            reporter_ids.append(default_reporter)
        
        return dataframes, reporter_ids

    def _process_multiple_csvs(self, csv_files: List[str], season_dir: str, timestamp_formats: List) -> Tuple[List[pd.DataFrame], List[str]]:
        """Process multiple CSV files, each representing one optimizer."""
        dataframes = []
        reporter_ids = []
        
        for file in csv_files:
            file_path = os.path.join(season_dir, file)
            df = pd.read_csv(file_path)
            
            # Extract reporter_id from filename
            reporter_id = file.split('_')[-1].split('.')[0]
            
            df_processed = self._process_dataframe(df, reporter_id, timestamp_formats)
            dataframes.append(df_processed)
            reporter_ids.append(reporter_id)
        
        return dataframes, reporter_ids

    def _process_dataframe(self, df: pd.DataFrame, reporter_id: str, timestamp_formats: List) -> pd.DataFrame:
        """Process individual dataframe with proper column renaming and timestamp parsing."""
        # Ensure first column is Timestamp
        if df.columns[0] != 'Timestamp':
            df.rename(columns={df.columns[0]: 'Timestamp'}, inplace=True)
        
        # Rename columns to include reporter_id
        rename_map = {
            'panel_current': f'panel_current_{reporter_id}',
            'panel_voltage': f'panel_voltage_{reporter_id}',
            'temperature': f'temperature_{reporter_id}',
            'panel_temperature': f'panel_temperature_{reporter_id}',
            'power': f'power_{reporter_id}'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Parse timestamps
        for fmt in timestamp_formats:
            try:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], format=fmt)
                break
            except (ValueError, TypeError):
                continue
        
        # Set timestamp as index and keep only relevant columns
        df.set_index('Timestamp', inplace=True)
        df = df[list(rename_map.values())]
        
        return df

    def _synchronize_and_merge_data(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Synchronize timestamps across all DataFrames and merge them."""
        if not dataframes:
            raise ValueError("No dataframes to synchronize")
        
        # Find overlapping time window
        earliest_timestamp = max([df.index[0] for df in dataframes])
        latest_timestamp = min([df.index[-1] for df in dataframes])
        
        # Create 5-minute frequency index
        new_index = pd.date_range(start=earliest_timestamp, end=latest_timestamp, freq='5min')
        
        # Reindex each DataFrame
        for i in range(len(dataframes)):
            for index in new_index:
                if index not in dataframes[i].index:
                    dataframes[i].loc[index] = np.nan
        
        # Merge all DataFrames
        merged_data = pd.concat(dataframes, axis=1)
        merged_data.reset_index(inplace=True)
        
        return merged_data

    def _substitute_temperature_data(self):
        """Replace panel temperature with ambient temperature if use_a_T is True."""
        self.logger.info("Substituting panel temperature with ambient temperature")
        
        # Remove panel_temperature columns
        cols_to_drop = [col for col in self.merged_data.columns if 'panel_temperature' in col]
        self.merged_data.drop(columns=cols_to_drop, inplace=True)
        
        # Create new panel_temperature columns from temperature columns
        for col in [col for col in self.merged_data.columns if 'temperature' in col and 'panel_temperature' not in col]:
            new_col = col.replace('temperature', 'panel_temperature')
            self.merged_data[new_col] = self.merged_data[col]

    def extract_module_parameters(self):
        """
        Read the PV module's electrical parameters from its corresponding .PAN file.
        
        Extracts:
        - Series resistance (RSerie)
        - Shunt resistance (RShunt)  
        - Number of cells in series (NCelS)
        - Ideality factor (Gamma)
        """
        self.logger.info("Extracting module parameters from .PAN file...")
        
        # Find site directory
        site_folders = [d for d in os.listdir(self.data_dir) if self.site_id in d]
        site_dir = os.path.join(self.data_dir, site_folders[0])
        
        # Find .PAN file
        pan_files = [f for f in os.listdir(site_dir) if f.endswith('.PAN')]
        if not pan_files:
            raise ValueError("No .PAN file found in site directory")
        
        pan_file_path = os.path.join(site_dir, pan_files[0])
        
        # Parse .PAN file
        with open(pan_file_path, 'r') as f:
            pan_data = f.readlines()
        
        # Extract parameters
        for line in pan_data:
            if 'RSerie' in line:
                self.module_params['series_resistance'] = float(line.split('=')[1].strip())
            elif 'RShunt' in line:
                self.module_params['shunt_resistance'] = float(line.split('=')[1].strip())
            elif 'NCelS' in line:
                self.module_params['num_cells_series'] = int(line.split('=')[1].strip())
            elif 'Gamma' in line:
                self.module_params['ideality_factor'] = float(line.split('=')[1].strip())
        
        # Validate that all required parameters were found
        required_params = ['series_resistance', 'shunt_resistance', 'num_cells_series', 'ideality_factor']
        missing_params = [p for p in required_params if p not in self.module_params]
        
        if missing_params:
            raise ValueError(f"Missing required parameters from .PAN file: {missing_params}")
        
        self.logger.info(f"Module parameters extracted: {self.module_params}")

    def run_analysis(self):
        """
        Main orchestrator for the analysis.
        
        Iterates through each timestamp and:
        1. Calculates I0 and IL for each module
        2. Reconstructs individual I-V curves
        3. Combines curves into series-connected string
        4. Calculates mismatch losses
        5. Generates visualizations
        """
        self.logger.info("Starting main analysis loop...")
        
        # Setup output directory
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        first_month = pd.to_datetime(self.merged_data['Timestamp'].iloc[0]).strftime('%B')
        output_dir = os.path.join(self.results_dir, f"v_from_i_combined\\{self.site_id}_{first_month}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # Filter data to specified number of days
        filtered_data = self._filter_data_by_days()
        
        # Initialize containers for GIF creation
        image_files = []
        
        self.logger.info(f"Processing {len(filtered_data)} timesteps...")
        
        # Main analysis loop
        for idx in range(len(filtered_data)):
            if idx % 100 == 0:  # Progress logging
                self.logger.info(f"Processing timestep {idx}/{len(filtered_data)}")
            
            current_timestamp = pd.to_datetime(filtered_data['Timestamp'].iloc[idx])
            
            # Skip if all power is zero
            if self._all_power_zero(filtered_data, idx):
                continue
            
            # Process this timestep
            image_path = self._process_timestep(filtered_data, idx, current_timestamp)
            if image_path:
                image_files.append(image_path)
        
        # Create GIF from all images
        self._create_gif(image_files)
        
        self.logger.info("Analysis loop completed")

    def _filter_data_by_days(self) -> pd.DataFrame:
        """Filter merged data to specified number of days."""
        start_date = pd.to_datetime(self.merged_data['Timestamp'].iloc[0])
        end_date = start_date + timedelta(days=self.num_days_to_plot)
        
        return self.merged_data[
            (pd.to_datetime(self.merged_data['Timestamp']) >= start_date) & 
            (pd.to_datetime(self.merged_data['Timestamp']) < end_date)
        ]

    def _all_power_zero(self, data: pd.DataFrame, idx: int) -> bool:
        """Check if all optimizers report zero power at this timestep."""
        return all(
            data.get(f'power_{optimiser}', pd.Series([0]*len(data))).iloc[idx] == 0
            or np.isnan(data.get(f'power_{optimiser}', pd.Series([0]*len(data))).iloc[idx])
            for optimiser in self.reporter_ids
        )

    def _process_timestep(self, data: pd.DataFrame, idx: int, current_timestamp: pd.Timestamp) -> Optional[str]:
        """Process a single timestep and generate visualization."""
        # Setup plotting
        fig_long, axs_long = plt.subplots(1, 3, figsize=self.long_hoz_figsize)
        self._setup_subplot_formatting(axs_long)
        
        # Initialize variables
        combined_voltage = np.zeros_like(self.currents)
        valid_data_found = False
        sum_iv = 0
        max_power = np.nan
        
        # Process each optimizer
        for optimiser in self.reporter_ids:
            voltage, iv_contribution, has_valid_data = self._process_optimizer(
                data, idx, optimiser, axs_long, current_timestamp
            )
            
            if has_valid_data:
                valid_data_found = True
                combined_voltage += voltage
                sum_iv += iv_contribution
        
        # Process combined IV curve if valid data exists
        if valid_data_found:
            max_power = self._process_combined_curve(combined_voltage, axs_long[2], current_timestamp)
        
        # Save results and generate plot
        self._save_timestep_results(current_timestamp, sum_iv)
        image_path = self._save_timestep_plot(fig_long, current_timestamp, sum_iv, max_power)
        
        plt.close(fig_long)
        return image_path

    def _setup_subplot_formatting(self, axs):
        """Setup formatting for the three subplots."""
        titles = ["Recorded MPP", "Reconstructed IV Curves", "Series combined IV Curve"]
        xlims = [self.x_limit_module, self.x_limit_module, self.x_limit_inverter]
        ylims = [self.y_limit_module, self.y_limit_module, self.y_limit_inverter]
        
        for i, (ax, title, xlim, ylim) in enumerate(zip(axs, titles, xlims, ylims)):
            ax.set_title(title, fontsize=self.title_size, pad=20)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel('Voltage (V)', fontsize=self.axis_label_size)
            ax.set_ylabel('Current (A)', fontsize=self.axis_label_size)
            ax.tick_params(axis='both', labelsize=self.axis_num_size)

    def _process_optimizer(self, data: pd.DataFrame, idx: int, optimiser: str, axs, current_timestamp: pd.Timestamp) -> Tuple[np.ndarray, float, bool]:
        """Process individual optimizer data and return voltage curve, IV contribution, and validity."""
        # Get optimizer data
        optimiser_voltage = data[f'panel_voltage_{optimiser}']
        optimiser_current = data[f'panel_current_{optimiser}']
        panel_temperature = data[f'panel_temperature_{optimiser}']
        
        # Check for valid data
        is_nan_or_zero = (
            optimiser_voltage.iloc[idx] == 0 or
            optimiser_current.iloc[idx] == 0 or
            np.isnan(optimiser_voltage.iloc[idx]) or
            np.isnan(optimiser_current.iloc[idx])
        )
        
        if is_nan_or_zero:
            # Plot no-data marker and record NaN results
            axs[0].plot(0, 0, 'kx', label=f'Opt {optimiser} (no data)')
            voltage = np.zeros_like(self.currents)
            axs[1].plot(voltage, self.currents, label=f'Opt {optimiser}')
            self._record_module_params(current_timestamp, optimiser, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            return voltage, 0.0, False
        
        # Valid data - extract values
        panel_voltage = optimiser_voltage.iloc[idx]
        panel_current = optimiser_current.iloc[idx]
        panel_temp_k = panel_temperature.iloc[idx] + 273.15
        
        # Plot recorded MPP
        axs[0].plot(panel_voltage, panel_current, 'ro', markersize=8, label=f'Opt {optimiser}')
        
        # Calculate diode parameters
        i0, il = self._calculate_i0_il(panel_current, panel_voltage, panel_temp_k)
        
        # Reconstruct IV curve
        voltage = self._reconstruct_iv_curve(i0, il, panel_temp_k)
        
        # Plot reconstructed curve
        axs[1].plot(voltage, self.currents, label=f'Opt {optimiser}')
        axs[1].plot(panel_voltage, panel_current, 'ro')
        
        # Calculate single diode parameters
        params_dict = self._get_single_diode_params(i0, il, panel_temp_k)
        self._record_module_params(current_timestamp, optimiser, i0, **params_dict)
        
        return voltage, panel_voltage * panel_current, True

    def _calculate_i0_il(self, panel_current: float, panel_voltage: float, panel_temperature: float) -> Tuple[float, float]:
        """Calculate diode saturation current (I0) and light-generated current (IL)."""
        # Calculate thermal voltage
        vth = (self.boltzmann_constant * panel_temperature / self.electron_charge 
               if self.use_dynamic_vth else 0.02569)  # 25°C thermal voltage
        
        # Extract module parameters
        Rs = self.module_params['series_resistance']
        Rsh = self.module_params['shunt_resistance']
        n = self.module_params['ideality_factor']
        N = self.module_params['num_cells_series']
        
        # Calculate I0
        exp_term = np.exp(-(panel_voltage + panel_current * Rs) / (n * N * vth))
        frac_term = n * N * vth / panel_voltage
        numerator = panel_current * (1 + Rs/Rsh) - panel_voltage/Rsh
        denominator = 1 - panel_current * Rs / panel_voltage
        i0 = numerator / denominator * frac_term * exp_term
        
        # Calculate IL
        first_term = panel_current * (1 + Rs/Rsh)
        second_term = panel_voltage / Rsh
        third_term = i0 * (np.exp((panel_voltage + panel_current * Rs) / (n * N * vth)) - 1)
        il = first_term + second_term + third_term
        
        return i0, il

    def _reconstruct_iv_curve(self, i0: float, il: float, panel_temperature: float) -> np.ndarray:
        """Reconstruct full IV curve from calculated parameters."""
        vth = (self.boltzmann_constant * panel_temperature / self.electron_charge 
               if self.use_dynamic_vth else 0.02569)
        
        params = {
            'photocurrent': il,
            'saturation_current': i0,
            'resistance_series': self.module_params['series_resistance'],
            'resistance_shunt': self.module_params['shunt_resistance'],
            'nNsVth': self.module_params['ideality_factor'] * self.module_params['num_cells_series'] * vth
        }
        
        voltage = pvlib.pvsystem.v_from_i(
            current=self.currents,
            photocurrent=params['photocurrent'],
            saturation_current=params['saturation_current'],
            resistance_series=params['resistance_series'],
            resistance_shunt=params['resistance_shunt'],
            nNsVth=params['nNsVth']
        )
        
        return voltage

    def _get_single_diode_params(self, i0: float, il: float, panel_temperature: float) -> Dict[str, float]:
        """Calculate single diode parameters (Isc, Voc, Pmp, etc.)."""
        vth = (self.boltzmann_constant * panel_temperature / self.electron_charge 
               if self.use_dynamic_vth else 0.02569)
        
        params = {
            'photocurrent': il,
            'saturation_current': i0,
            'resistance_series': self.module_params['series_resistance'],
            'resistance_shunt': self.module_params['shunt_resistance'],
            'nNsVth': self.module_params['ideality_factor'] * self.module_params['num_cells_series'] * vth
        }
        
        results = pvlib.pvsystem.singlediode(**params)
        
        isc = results['i_sc']
        voc = results['v_oc']
        pmp = results['p_mp']
        imp = results['i_mp']
        vmp = results['v_mp']
        ff = (pmp / (isc * voc)) if (isc > 0 and voc > 0) else np.nan
        
        return {
            'isc': isc, 'voc': voc, 'ff': ff, 'pmp': pmp, 'imp': imp, 'vmp': vmp
        }

    def _record_module_params(self, timestamp: pd.Timestamp, optimizer: str, i0: float, 
                            isc: float, voc: float, ff: float, pmp: float, imp: float, vmp: float):
        """Record module parameters for this timestep and optimizer."""
        new_row = pd.DataFrame({
            'Timestamp': [timestamp],
            'Optimizer': [optimizer],
            'I0': [i0],
            'Isc': [isc],
            'Voc': [voc],
            'FF': [ff],
            'Pmp': [pmp],
            'Imp': [imp],
            'Vmp': [vmp]
        })
        self.module_param_df = pd.concat([self.module_param_df, new_row], ignore_index=True)

    def _process_combined_curve(self, combined_voltage: np.ndarray, ax, current_timestamp: pd.Timestamp) -> float:
        """Process the series-combined IV curve and extract MPP."""
        # Calculate power and find MPP
        power = combined_voltage * self.currents
        max_power_idx = np.argmax(power)
        max_voltage = combined_voltage[max_power_idx]
        max_current = self.currents[max_power_idx]
        max_power = power[max_power_idx]
        
        # Find Isc and Voc
        isc_combined = self.currents[np.where(combined_voltage > 0)[0][-1]] if np.any(combined_voltage > 0) else 0
        voc_combined = combined_voltage[np.where(self.currents == 0)[0][0]] if len(np.where(self.currents == 0)[0]) > 0 else 0
        
        # Plot combined curve
        ax.plot(combined_voltage, self.currents, label='Combined IV Curve')
        ax.plot(max_voltage, max_current, 'ro', label='Max Power Point')
        ax.plot(voc_combined, 0, 'go', label='Voc')
        ax.plot(0, isc_combined, 'bo', label='Isc')
        
        # Record results
        max_power_point = pd.DataFrame({
            'Timestamp': [current_timestamp],
            'Max Voltage (V)': [max_voltage],
            'Max Current (A)': [max_current],
            'Max Power (W)': [max_power],
            'Voc (V)': [voc_combined],
            'Isc (A)': [isc_combined]
        })
        self.max_power_df_combined = pd.concat([self.max_power_df_combined, max_power_point], ignore_index=True)
        
        # Record Pmppt data
        pmppt_point = pd.DataFrame({
            'Timestamp': [current_timestamp], 
            'Pmppt (W)': [max_power]
        })
        self.pmppt_data = pd.concat([self.pmppt_data, pmppt_point], ignore_index=True)
        
        return max_power

    def _save_timestep_results(self, timestamp: pd.Timestamp, sum_iv: float):
        """Save results for this timestep."""
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        iv_sum_point = pd.DataFrame({
            'Timestamp': [timestamp_str], 
            'Sum of I*V (W)': [sum_iv]
        })
        self.iv_sum_data = pd.concat([self.iv_sum_data, iv_sum_point], ignore_index=True)

    def _save_timestep_plot(self, fig, timestamp: pd.Timestamp, sum_iv: float, max_power: float) -> str:
        """Save the timestep plot and return the file path."""
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # Calculate percentage difference
        percentage_diff = 0
        if sum_iv != 0:
            percentage_diff = ((sum_iv - max_power) / sum_iv) * 100
        
        # Create title
        title_row1 = f"Site: {self.site_id} | {timestamp_str}"
        title_row2 = f"Sum of MPP: {sum_iv:.2f} W | Combined IV MPP: {max_power:.2f} W"
        title_row3 = f"% Diff: {percentage_diff:.2f}%"
        fig.suptitle(f"{title_row1}\n{title_row2}\n{title_row3}", fontsize=self.title_size)
        
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        
        # Save plot
        file_path = os.path.join(self.output_dir, 
                                f'long_horizontal_{timestamp_str.replace(":", "-").replace(" ", "_")}.png')
        plt.savefig(file_path, bbox_inches='tight')
        
        return file_path

    def _create_gif(self, image_files: List[str]):
        """Create GIF from all timestep plots."""
        if not image_files:
            self.logger.warning("No images to create GIF")
            return
        
        gif_path = os.path.join(self.output_dir, 'combined_iv_curves_long.gif')
        with imageio.get_writer(gif_path, mode='I', duration=200, loop=0) as writer:
            for filename in image_files:
                image = imageio.imread(filename)
                writer.append_data(image)
        
        self.logger.info(f"GIF saved as: {gif_path}")

    def generate_plots(self):
        """Generate summary plots after analysis is complete."""
        self.logger.info("Generating summary plots...")
        
        # Create combined data
        combined_data = self._create_combined_data()
        
        # Generate power comparison plot
        self._generate_power_comparison_plot(combined_data)
        
        # Generate percentage difference plot
        self._generate_percentage_difference_plot(combined_data)
        
        self.logger.info("Summary plots generated")

    def _create_combined_data(self) -> pd.DataFrame:
        """Create combined dataset with mismatch calculations."""
        combined_data = pd.concat([self.iv_sum_data, self.pmppt_data['Pmppt (W)']], axis=1)
        combined_data['Season'] = self.season
        combined_data['Site ID'] = self.site_id
        combined_data['Timestamp'] = pd.to_datetime(combined_data['Timestamp'])
        return combined_data

    def _generate_power_comparison_plot(self, combined_data: pd.DataFrame):
        """Generate plot comparing Pmppt vs Sum of I*V."""
        fig, ax = plt.subplots(figsize=self.long_hoz_figsize)
        
        ax.plot(combined_data['Timestamp'], combined_data['Pmppt (W)'], 
                label='Series connection', alpha=0.4)
        ax.plot(combined_data['Timestamp'], combined_data['Sum of I*V (W)'], 
                label='Sum of MPP', alpha=0.4)
        
        ax.set_xlabel('Time', fontsize=self.axis_label_size)
        ax.set_ylabel('Power (W)', fontsize=self.axis_label_size)
        
        # Calculate mismatch
        sum_iv_E = combined_data['Sum of I*V (W)'].sum()
        pmppt_E = combined_data['Pmppt (W)'].sum()
        sum_mismatch = (sum_iv_E - pmppt_E) / sum_iv_E if sum_iv_E > 0 else 0
        
        first_month = combined_data['Timestamp'].iloc[0].strftime('%B')
        ax.set_title(f'Site ID: {self.site_id}, Month: {first_month}\nMismatch: {sum_mismatch * 100:.2f}%',
                    fontsize=self.title_size, pad=20)
        
        ax.legend(loc='upper right', fontsize=self.axis_num_size-5)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.tick_params(axis='x', labelsize=self.axis_num_size)
        ax.tick_params(axis='y', labelsize=self.axis_num_size)
        
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.20)
        
        plot_file = os.path.join(self.output_dir, 'pmppt_vs_sum_iv.png')
        fig.savefig(plot_file, dpi=300)
        plt.close(fig)
        
        self.logger.info(f"Power comparison plot saved: {plot_file}")
        self.logger.info(f"Calculated mismatch: {sum_mismatch * 100:.2f}%")

    def _generate_percentage_difference_plot(self, combined_data: pd.DataFrame):
        """Generate plot showing percentage difference over time."""
        fig, ax = plt.subplots(figsize=self.long_hoz_figsize)
        
        # Calculate differences
        percentage_diff = ((combined_data['Sum of I*V (W)'] - combined_data['Pmppt (W)']) / 
                          combined_data['Sum of I*V (W)']) * 100
        percentage_diff[combined_data['Sum of I*V (W)'] < 1] = 0
        abs_diff = (combined_data['Sum of I*V (W)'] - combined_data['Pmppt (W)']).abs()
        
        # Plot percentage difference
        ax.plot(combined_data['Timestamp'], percentage_diff, 
                label='Percentage Difference (%)', alpha=0.4, color='orange')
        ax.set_xlabel('Time', fontsize=self.axis_label_size)
        ax.set_ylabel('Percentage Difference (%)', fontsize=self.axis_label_size)
        
        # Secondary y-axis for absolute difference
        ax2 = ax.twinx()
        ax2.plot(combined_data['Timestamp'], abs_diff, 
                label='Absolute Difference (W)', alpha=0.4, color='blue')
        ax2.set_ylabel('Absolute Difference (W)', fontsize=self.axis_label_size)
        
        # Title and legend
        sum_iv_E = combined_data['Sum of I*V (W)'].sum()
        pmppt_E = combined_data['Pmppt (W)'].sum()
        sum_mismatch = (sum_iv_E - pmppt_E) / sum_iv_E if sum_iv_E > 0 else 0
        
        first_month = combined_data['Timestamp'].iloc[0].strftime('%B')
        ax.set_title(f'Site ID: {self.site_id}, Month: {first_month}\nMismatch: {sum_mismatch * 100:.2f}%',
                    fontsize=self.title_size, pad=20)
        
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=self.axis_num_size-5)
        
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.tick_params(axis='x', labelsize=self.axis_num_size)
        ax.tick_params(axis='y', labelsize=self.axis_num_size)
        ax2.tick_params(axis='y', labelsize=self.axis_num_size)
        
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.20)
        
        plot_file = os.path.join(self.output_dir, 'percentage_difference.png')
        fig.savefig(plot_file, dpi=300)
        plt.close(fig)
        
        self.logger.info(f"Percentage difference plot saved: {plot_file}")

    def save_results(self):
        """Save all tabular results to files."""
        self.logger.info("Saving results to files...")
        
        # Create combined data
        combined_data = self._create_combined_data()
        
        # Save individual result files
        excel_paths = [
            (os.path.join(self.output_dir, 'iv_sum_data.xlsx'), self.iv_sum_data),
            (os.path.join(self.output_dir, 'pmppt_data.xlsx'), self.pmppt_data),
            (os.path.join(self.output_dir, f'combined_data_{self.season}_{self.site_id}.xlsx'), combined_data)
        ]
        
        for path, data in excel_paths:
            data.to_excel(path, index=False)
            self.logger.info(f"Saved: {path}")
        
        # Save module parameters as CSV
        params_csv = os.path.join(self.output_dir, 'module_param_df.csv')
        self.module_param_df.to_csv(params_csv, index=False)
        self.logger.info(f"Saved: {params_csv}")

    def calculate_mismatch_loss(self) -> float:
        """Calculate overall mismatch loss percentage."""
        combined_data = self._create_combined_data()
        
        sum_iv_E = combined_data['Sum of I*V (W)'].sum()
        pmppt_E = combined_data['Pmppt (W)'].sum()
        
        if sum_iv_E > 0:
            mismatch_loss = (sum_iv_E - pmppt_E) / sum_iv_E * 100
        else:
            mismatch_loss = 0.0
        
        self.logger.info(f"Total mismatch loss: {mismatch_loss:.2f}%")
        return mismatch_loss

    def get_results_summary(self) -> Dict[str, any]:
        """Get a summary of analysis results."""
        combined_data = self._create_combined_data()
        
        return {
            'site_id': self.site_id,
            'season': self.season,
            'num_optimizers': len(self.reporter_ids),
            'num_timesteps': len(combined_data),
            'mismatch_loss_percent': self.calculate_mismatch_loss(),
            'total_sum_mpp_energy': combined_data['Sum of I*V (W)'].sum(),
            'total_series_energy': combined_data['Pmppt (W)'].sum(),
            'analysis_period_days': self.num_days_to_plot,
            'output_directory': getattr(self, 'output_dir', None)
        }