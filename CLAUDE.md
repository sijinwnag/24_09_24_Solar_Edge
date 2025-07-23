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

### 2. Mismatch Results Analysis (`25_07_22_Mismatch_results_analyser.ipynb`)
Comprehensive analysis of generated results:
- Aggregates mismatch data across sites and seasons
- Performs outlier detection for Voc/Isc to identify bypass diode activation
- Analyzes correlation with climate zones, latitude, system size, and shading
- Creates comparative visualizations before/after filtering diode activation events

### 3. Daily Mismatch Analysis (Section 6 of Analyzer Notebook)
Daily analysis functionality integrated within the main analyzer notebook:
- Available as Section 6 in `25_07_22_Mismatch_results_analyser.ipynb`
- Processes combined "no_diode" filtered data files to extract daily mismatch metrics
- Generates daily summary Excel file (daily_mismatch_summary.xlsx) with 20 columns
- Streamlined for core Excel output without visualization or caching overhead
- Significantly faster execution than seasonal analysis for large datasets

### 4. LTSpice Bypass Diode Simulation Analysis (`25_04_03_module_diode_activation_LTSpice/`)
Physics-based circuit simulation for bypass diode validation:
- **LTSpice Circuit Files**: `.asc` files modeling 72-cell module with bypass diodes based on Bomen Solar Farm specifications
- **Simulation Data**: `25_04_04_bypass.xlsx` containing 4,851 I-V data points for normal/1-diode/2-diode scenarios (0-48.5V range)
- **Primary Analysis Tool**: `25_07_22_PV_IV_Curve_Analysis.ipynb` - Comprehensive analysis with:
  - Robust Excel loading (handles OneDrive permission issues using temporary file method)
  - Maximum Power Point (MPP) detection and analysis
  - Power loss quantification (validates 34.7% loss for 1 diode, 41.8% for 2 diodes)
  - Publication-quality plotting with consistent formatting parameters
- **Research Integration**: Validates theoretical bypass diode effects against empirical SolarEdge data
- **Key Findings**: Progressive power degradation, voltage shift patterns, fill factor impact

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

## Commands and Execution

**Running Analysis Scripts:**
```bash
# Run main Jupyter notebooks (requires Jupyter installation)
jupyter notebook Code/25_04_27_Mismatch_results_generator.ipynb
jupyter notebook Code/25_07_22_Mismatch_results_analyser.ipynb

# Run LTSpice simulation analysis
jupyter notebook Code/25_04_03_module_diode_activation_LTSpice/25_07_22_PV_IV_Curve_Analysis.ipynb

# For development environment focusing on bypass diode analysis
cd Code/25_04_03_module_diode_activation_LTSpice/
jupyter lab 25_07_22_PV_IV_Curve_Analysis.ipynb

# Start Jupyter Lab for better notebook experience
jupyter lab

# Install required dependencies (if not present)
pip install pandas numpy matplotlib scipy pvlib geopy imageio openpyxl xlrd
pip install kgcpy  # Köppen-Geiger climate classification

# Additional dependencies for robust Excel handling
pip install openpyxl xlrd  # Multiple Excel engines for compatibility
```

**Development Environment Setup:**
```bash
# Optional: Create virtual environment
python -m venv solar_analysis_env
source solar_analysis_env/bin/activate  # Linux/Mac
# or
solar_analysis_env\Scripts\activate     # Windows

# Install requirements
pip install jupyter pandas numpy matplotlib scipy pvlib geopy imageio kgcpy
```

**File Management Commands:**
```bash
# Navigate to project directory
cd "C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\24_09_24_Solar_Edge"

# Check data structure
ls Data/           # List all site folders
ls Results/        # List analysis results

# Monitor analysis progress (for long-running scripts)
python Code/25_01_22_Daily_mismatch_analyser.py | tee analysis.log
```

## Dependencies and Environment

**Required Python Libraries:**
- Core: `pandas`, `numpy`, `matplotlib`, `scipy`
- Solar modeling: `pvlib`
- Geographic: `geopy`, `kgcpy` (Köppen-Geiger climate classification) 
- Data processing: `imageio` (for GIF generation)
- Excel handling: `openpyxl`, `xlrd` (multiple engines for robust file access)
- Standard: `os`, `sys`, `datetime`, `json`, `requests`, `pathlib`, `shutil`, `tempfile`

**Development Environment:**
- Jupyter notebooks are the primary development interface
- All notebooks should be run in the same Python environment
- No virtual environment configuration files present - use system Python or manually create environment

## Common Development Tasks

**Running Complete Analysis:**
1. Execute `25_04_27_Mismatch_results_generator.ipynb` to process raw data
2. Execute `25_07_22_Mismatch_results_analyser.ipynb` to analyze results (includes Section 6 for daily analysis)
3. Results automatically saved to timestamped folders in `Results/v_from_i_combined/`

**Daily Analysis Workflow:**
- Use Section 6 of `25_07_22_Mismatch_results_analyser.ipynb` for daily analysis
- Implements daily-level granularity instead of seasonal aggregation
- Output folder: `Results/daily_analysis_results/` (contains daily_mismatch_summary.xlsx)
- Requires "no_diode" filtered combined data files from generator notebook
- Includes optimizations: vectorized operations, caching, reduced plotting

**Processing New Site Data:**
1. Add site folder to `Data/` with proper structure
2. Update `site_ids` list in generator notebook (cell defining `site_ids = ['4111846']`)
3. Ensure .PAN file contains required parameters (RSerie, RShunt, NCelS, Gamma)

**Configuration Variables (Generator Notebook):**
- **Directory Paths**: `data_dir`, `base_dir`, `results_dir`, `summary_dir`
- **Plot Limits**: `y_limit_module` (0,15), `x_limit_module` (0,60), `y_limit_inverter` (0,17), `x_limit_inverter` (0,1200)
- **Temperature Settings**: `use_dynamic_vth` (thermal voltage calculation), `use_a_T` (ambient temperature substitution)
- **Data Processing**: `num_days_to_plot` (default 10), `site_ids` list for target sites

**Standardized Plotting Parameters (Applied Across All Notebooks):**
- **Font Sizes**: `axis_label_size` (20), `axis_num_size` (20), `title_size` (22), `text_size` (20)
- **Figure Sizes**: `figure_size` (6,6), `long_hoz_figsize` (12,6), `two_by_two_figsize` (12,12)
- **Publication Quality**: All notebooks use consistent formatting for professional output
- **LTSpice Enhanced**: Uses larger parameters for bypass diode analysis visualization

**Seasonal Analysis:**
- Northern/Southern hemisphere seasons mapped automatically based on country
- Month-to-season mapping handles seasonal data organization
- Climate zone lookup uses geopy + Köppen-Geiger classification

## Analysis Pipeline Architecture

**Data Flow (Multi-File Architecture):**
```
Raw Telemetry Data (Data/{site_id}/) 
    ↓
Generator Notebook (25_04_27_*) → Processes I-V data → Results/v_from_i_combined/
    ↓
Analyzer Notebook (25_07_22_*) → Aggregates & analyzes → Statistical outputs
    ├── Section 6: Daily analysis → Results/daily_analysis_results/

LTSpice Simulation Data (25_04_04_bypass.xlsx)
    ↓
PV I-V Analysis (25_07_22_PV_IV_Curve_Analysis.ipynb) → Bypass diode validation
    ↓ 
Validates theoretical predictions (34.7%/41.8% power losses) → Integration with empirical analysis
```

**Cross-File Dependencies:**
- Generator notebook outputs become inputs for analyzer notebook  
- Daily analysis (Section 6) requires "no_diode" filtered files from generator
- Site summary Excel file (`25_05_01_Newsites_summary.xlsx`) used across all workflows
- Climate zone data cached and reused between analysis runs

**Shared Analysis Components:**
- Single-diode model parameters (.PAN files) used consistently across all analyses
- FFT calculations implemented differently: full computation (generator) vs. optimized estimation (daily analysis)
- Outlier detection algorithms shared between analyzer notebook and daily analysis section
- Geocoding and climate classification standardized across workflows

## Key Functions and Analysis Logic

**Core Architecture Components:**

**I-V Curve Reconstruction:**
- `I0()` and `IL()` functions calculate diode model parameters from MPP data
- `pvlib.pvsystem.v_from_i()` generates full I-V curves using single-diode model
- Series combination sums voltages at constant current for system-level analysis

**Data Processing Pipeline:**
1. **Data Loading**: Multiple CSV format support with automatic timestamp synchronization
2. **Parameter Extraction**: .PAN file parsing for module-specific parameters (Rs, Rsh, n, Ncells)
3. **I-V Reconstruction**: Single-diode model parameter calculation for each optimizer
4. **Series Combination**: Voltage summation at constant current to create system I-V curve
5. **Mismatch Calculation**: Power comparison between sum-of-MPP and series-connection

**Mismatch Loss Calculation:**
```
Mismatch Loss (%) = (Sum_of_MPP - Series_MPP) / Sum_of_MPP × 100
```

**Data Quality Filtering:**
- Removes timestamps where all optimizers report zero power
- Filters out diode activation events for "true" mismatch analysis  
- Handles missing data with appropriate NaN treatment
- Temperature sensor validation and optional ambient temperature substitution (`use_a_T = True`)

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
- **Excel Permission Issues**: LTSpice notebook includes robust loading with temporary file method for OneDrive/permission problems
- **Variable Name Errors**: Historical `mmp_*` vs `mpp_*` naming issues have been resolved in all analysis notebooks

**File Structure Requirements:**
- Each site must have properly named seasonal/monthly subfolders
- Optimizer data files must contain 'optimizer_data' in filename
- Combined data files for analysis must contain both 'combined_data' and 'no_diode' in filename
- Results folders use timestamp-based naming: `{site_id}_{season}_{YYYYMMDD_HHMMSS}`

**Performance Considerations:**
- Large datasets may require substantial memory for FFT analysis and plotting
- GIF generation can be time-intensive for long time series
- Climate zone lookup requires internet connectivity for geopy geocoding
- Daily analysis (Section 6) includes optimizations: vectorized operations, caching, reduced plotting
- For very large datasets, consider using `num_days_to_plot` parameter to limit processing time
- Plotting parameters can be adjusted for performance: `figure_size`, `y_limit_module`, `x_limit_module`

## Testing and Validation

**Data Validation Procedures:**
- Cross-validation between seasonal aggregation and daily analysis results
- Comparison of mismatch calculations before/after diode activation filtering
- Verification that sum-of-MPP always exceeds series-connection power (physical constraint)
- FFT period calculations validated against expected diurnal patterns

**Output Validation:**
- Excel exports checked for data completeness and format consistency
- Plot generation verified across different data sizes and site configurations
- Climate zone assignments cross-checked with geographic coordinates
- Statistical summaries validated against manual calculations for sample datasets

**Performance Validation:**
- Daily analysis section timing optimizations within notebook execution
- Memory usage monitoring for large dataset processing
- Geocoding cache effectiveness measured by API call reduction
- I-V curve reconstruction accuracy validated against measured MPP data

## Code Quality and Recent Improvements

**Resolved Issues (July 2025):**
- **Variable Naming Consistency**: All `mmp_*` variable references corrected to `mpp_*` across notebooks
- **Excel Loading Robustness**: LTSpice notebook now handles OneDrive permission issues with temporary file method
- **Plot Formatting Standardization**: Enhanced formatting parameters applied for publication-quality output
- **Error Handling**: Comprehensive error handling for missing files, corrupted data, and permission issues

**Backup Strategy:**
- Critical notebook backups created before major modifications (e.g., `25_07_22_PV_IV_Curve_Analysis_backup.ipynb`)
- Version control through file naming with dates for analysis notebooks

**Testing and Validation:**
- All key analysis functions tested with synthetic and real data
- Cross-validation between theoretical LTSpice predictions and empirical results
- Data loading robustness validated across different file systems and permission scenarios

## Notes

- All timestamps synchronized to 5-minute intervals across optimizers
- Temperature-dependent thermal voltage calculation for accuracy
- Extensive data validation and outlier detection for research quality
- Results exported in both Excel and CSV formats for further analysis
- Plotting parameters standardized for publication-quality figures