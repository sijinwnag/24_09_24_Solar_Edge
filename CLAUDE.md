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

### 3. Daily Mismatch Analysis (`25_01_22_Daily_mismatch_analyser.py`)
Optimized Python script for daily-granularity analysis:
- Processes combined data files to extract daily mismatch metrics
- Vectorized FFT calculations for improved performance
- Cached geocoding to avoid repeated API calls
- Generates daily summary Excel file and essential visualizations
- Significantly faster execution than seasonal analysis for large datasets

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
# Execute daily mismatch analysis (optimized)
python Code/25_01_22_Daily_mismatch_analyser.py

# Run Jupyter notebooks (requires Jupyter installation)
jupyter notebook Code/25_04_27_Mismatch_results_generator.ipynb
jupyter notebook Code/25_05_01_Mismatch_results_analyser.ipynb

# Install required dependencies (if not present)
pip install pandas numpy matplotlib scipy pvlib geopy imageio
pip install kgcpy  # Köppen-Geiger climate classification
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

**Daily Analysis Workflow:**
- Execute `25_01_22_Daily_mismatch_analyser.py` directly or import functions into notebooks
- Implements daily-level granularity instead of seasonal aggregation
- Optimized for performance with vectorized operations and caching
- Output folder: `Results/daily_analysis_results/`
- Requires same input data structure but processes at daily intervals
- See `daily_mismatch_analysis_instructions.md` for detailed specifications

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

## Analysis Pipeline Architecture

**Data Flow (Multi-File Architecture):**
```
Raw Telemetry Data (Data/{site_id}/) 
    ↓
Generator Notebook (25_04_27_*) → Processes I-V data → Results/v_from_i_combined/
    ↓
Analyzer Notebook (25_05_01_*) → Aggregates & analyzes → Statistical outputs
    ↓
Daily Script (25_01_22_*) → Daily granularity → Results/daily_analysis_results/
```

**Cross-File Dependencies:**
- Generator notebook outputs become inputs for analyzer notebook
- Daily script requires "no_diode" filtered files from generator
- Site summary Excel file (`25_05_01_Newsites_summary.xlsx`) used across all workflows
- Climate zone data cached and reused between analysis runs

**Shared Analysis Components:**
- Single-diode model parameters (.PAN files) used consistently across all analyses
- FFT calculations implemented differently: full computation (generator) vs. optimized estimation (daily)
- Outlier detection algorithms shared between analyzer notebook and daily script
- Geocoding and climate classification standardized across workflows

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
- Daily analysis script includes optimizations: vectorized operations, caching, reduced plotting
- For very large datasets, consider using `SAMPLE_DAYS_LIMIT` parameter to limit processing

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
- Daily analysis script timing compared against notebook execution
- Memory usage monitoring for large dataset processing
- Geocoding cache effectiveness measured by API call reduction

## Notes

- All timestamps synchronized to 5-minute intervals across optimizers
- Temperature-dependent thermal voltage calculation for accuracy
- Extensive data validation and outlier detection for research quality
- Results exported in both Excel and CSV formats for further analysis
- Plotting parameters standardized for publication-quality figures