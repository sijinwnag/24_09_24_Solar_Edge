# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a solar photovoltaic (PV) research project analyzing mismatch losses in SolarEdge systems. The project uses telemetry data from real solar installations to study power losses caused by module-level mismatches and bypass diode activation. The analysis combines theoretical modeling with empirical data to quantify mismatch losses across different seasons, climates, and system configurations.

## Repository Structure

- **`Code/`** - Jupyter notebooks containing the main analysis workflows
- **`Data/`** - Solar telemetry data organized by site ID and season/month
  - Each site folder (e.g., `3455043/`) contains module data files (.PAN) and seasonal subfolders
  - Seasonal folders contain: `inverter_data.csv`, `optimizer_data.csv`, `site_daily_data.csv`
- **`Results/`** - Analysis outputs organized by methodology and date
  - `v_from_i_combined/` - Main results from voltage-from-current analysis
  - `2_diode_models/` - Two-diode model simulation results
- **`Paper/`** - Research papers and documentation
- **`old_data/`** - Legacy data from earlier analysis phases

## Key Analysis Workflows

### 1. Mismatch Results Generation (`25_04_27_Mismatch_results_generator.ipynb`)
Primary workflow for processing raw telemetry data:
- Loads optimizer-level current/voltage data from multiple sites
- Reconstructs I-V curves using single-diode model parameters from .PAN files
- Calculates series-combined system I-V curves
- Compares sum-of-MPP vs series-connection power to quantify mismatch losses
- Generates time-series plots and exports results to Excel

### 2. Mismatch Results Analysis (`25_05_01_Mismatch_results_analyser.ipynb`)
Comprehensive analysis of generated results:
- Aggregates mismatch data across sites and seasons
- Performs outlier detection for Voc/Isc to identify bypass diode activation
- Analyzes correlation with climate zones, latitude, system size, and shading
- Creates comparative visualizations before/after filtering diode activation events

## Data Structure

**Site Data Organization:**
```
Data/{site_id}/
├── {site_id} - {module_info}.PAN     # Module parameters
├── inverter_metadata.csv
├── optimizer_metadata.csv
└── {season_or_month}/
    ├── inverter_data.csv             # System-level power output
    ├── optimizer_data.csv            # Module-level I/V/T data  
    └── site_daily_data.csv
```

**Key Data Fields:**
- `panel_current`, `panel_voltage` - MPP operating point
- `panel_temperature`, `temperature` - Module and ambient temperature
- `power` - Measured MPP power output
- `reporter_id` - Unique optimizer identifier

## Analysis Parameters

**Outlier Detection (Diode Activation):**
- Voc outliers: IQR factor 1.5 for high, percentile-based for diode losses
- Isc outliers: IQR factor 1.5 (low) and 3.0 (high) with minimum IQR enforcement
- Diode activation conditions: Voc loss without Isc change, or Voc high + Isc low

**Physical Constants:**
- Single-diode model parameters extracted from .PAN files (Rs, Rsh, n, Ncells)
- Dynamic thermal voltage calculation based on module temperature
- Option to substitute ambient temperature for module temperature (`use_a_T = True`)

## Dependencies and Environment

**Required Python Libraries:**
- Core: `pandas`, `numpy`, `matplotlib`, `scipy`
- Solar modeling: `pvlib`
- Geographic: `geopy`, `kgcpy` (Köppen-Geiger climate classification)
- Data processing: `imageio` (for GIF generation)
- Standard: `os`, `sys`, `datetime`, `json`, `requests`

**Development Environment:**
- Jupyter notebooks are the primary development interface
- All notebooks should be run in the same Python environment
- No virtual environment configuration files present - use system Python or manually create environment

## Common Development Tasks

**Running Complete Analysis:**
1. Execute `25_04_27_Mismatch_results_generator.ipynb` to process raw data
2. Execute `25_05_01_Mismatch_results_analyser.ipynb` to analyze results
3. Results automatically saved to timestamped folders in `Results/v_from_i_combined/`

**Daily Analysis (New Workflow):**
- See `daily_mismatch_analysis_instructions.md` for detailed specifications
- Implements daily-level granularity instead of seasonal aggregation
- Output folder: `Results/daily_analysis_results/`
- Requires same input data structure but processes at daily intervals

**Processing New Site Data:**
1. Add site folder to `Data/` with proper structure
2. Update `site_ids` list in generator notebook (cell defining `site_ids = ['4111846']`)
3. Ensure .PAN file contains required parameters (RSerie, RShunt, NCelS, Gamma)

**Configuration Variables (Generator Notebook):**
- Modify `data_dir`, `base_dir`, `summary_dir` paths as needed
- Adjust plot limits: `y_limit_module`, `x_limit_module`, `y_limit_inverter`, `x_limit_inverter`
- Toggle temperature calculation: `use_dynamic_vth`, `use_a_T`
- Plotting parameters: `axis_label_size`, `title_size`, `figure_size`

**Seasonal Analysis:**
- Northern/Southern hemisphere seasons mapped automatically based on country
- Month-to-season mapping handles seasonal data organization
- Climate zone lookup uses geopy + Köppen-Geiger classification

## Key Functions and Analysis Logic

**I-V Curve Reconstruction:**
- `I0()` and `IL()` functions calculate diode model parameters from MPP data
- `pvlib.pvsystem.v_from_i()` generates full I-V curves
- Series combination sums voltages at constant current

**Mismatch Calculation:**
```
Mismatch Loss (%) = (Sum_of_MPP - Series_MPP) / Sum_of_MPP × 100
```

**Data Quality Filtering:**
- Removes timestamps where all optimizers report zero power
- Filters out diode activation events for "true" mismatch analysis
- Handles missing data with appropriate NaN treatment

## Results and Visualization

**Generated Outputs:**
- Time-series plots of individual I-V curves and series combination
- Animated GIFs showing I-V evolution over time
- Statistical summaries by climate zone, season, orientation, shading
- Comparative analysis before/after diode activation filtering
- Correlation analysis with latitude, system size, module count

**Key Findings Visualization:**
- Mismatch loss distributions and seasonal patterns
- Climate zone and geographic correlations
- Impact of shading and multi-orientation configurations
- Quantification of diode activation frequency and impact

## Troubleshooting and Data Quality

**Common Data Issues:**
- Missing or corrupted .PAN files: Ensure all required parameters (RSerie, RShunt, NCelS, Gamma) are present
- Inconsistent timestamp formats: Multiple formats supported in data loading (`%Y-%m-%d %H:%M:%S`, `%d/%m/%Y %H:%M`, etc.)
- Zero power readings: Automatically filtered out in analysis (all optimizers reporting zero power)
- Temperature sensor issues: Use `use_a_T = True` to substitute ambient for module temperature if module readings seem unrealistic

**File Structure Requirements:**
- Each site must have properly named seasonal/monthly subfolders
- Optimizer data files must contain 'optimizer_data' in filename
- Combined data files for analysis must contain both 'combined_data' and 'no_diode' in filename
- Results folders use timestamp-based naming: `{site_id}_{season}_{YYYYMMDD_HHMMSS}`

**Performance Considerations:**
- Large datasets may require substantial memory for FFT analysis and plotting
- GIF generation can be time-intensive for long time series
- Climate zone lookup requires internet connectivity for geopy geocoding

## Notes

- All timestamps synchronized to 5-minute intervals across optimizers
- Temperature-dependent thermal voltage calculation for accuracy
- Extensive data validation and outlier detection for research quality
- Results exported in both Excel and CSV formats for further analysis
- Plotting parameters standardized for publication-quality figures