# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a solar photovoltaic (PV) research project analyzing mismatch losses in SolarEdge systems. The project uses telemetry data from real solar installations to study power losses caused by module-level mismatches and bypass diode activation. The analysis combines theoretical modeling with empirical data to quantify mismatch losses across different seasons, climates, and system configurations.

**Core Research Question**: How do module-level mismatches and bypass diode activation affect overall system power output in real-world SolarEdge installations?

**Methodology**: Reconstruct I-V curves from MPP telemetry data using single-diode models, then compare sum-of-MPP vs. series-connection power to quantify mismatch losses.

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
- `panel_current`, `panel_voltage` - MPP operating point from SolarEdge optimizers
- `panel_temperature`, `temperature` - Module and ambient temperature (°C)
- `power` - Measured MPP power output (W)
- `reporter_id` - Unique optimizer identifier for tracking individual modules

**PAN File Format (.PAN files contain module specifications):**
```
PVObject_=pvModule
NCelS=66                    # Number of cells in series
RSerie=0.209               # Series resistance (Ω) 
RShunt=700                 # Shunt resistance (Ω)
Gamma=0.976                # Ideality factor (dimensionless)
Isc=11.160                 # Short-circuit current (A)
Voc=45.06                  # Open-circuit voltage (V)
Imp=10.640                 # MPP current (A)  
Vmp=37.59                  # MPP voltage (V)
PNom=400.0                 # Nominal power (W)
```

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

**Primary Analysis Workflow:**
```bash
# Navigate to project root
cd "C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\24_09_24_Solar_Edge"

# 1. Generate mismatch results from raw telemetry data
jupyter notebook Code/25_04_27_Mismatch_results_generator.ipynb

# 2. Analyze generated results with statistical analysis
jupyter notebook Code/25_07_22_Mismatch_results_analyser.ipynb

# 3. (Optional) Validate theoretical predictions with LTSpice simulation
jupyter notebook Code/25_04_03_module_diode_activation_LTSpice/25_07_22_PV_IV_Curve_Analysis.ipynb
```

**Development Environment:**
```bash
# Start Jupyter Lab (recommended for development)
jupyter lab

# Alternative: Start Jupyter Notebook
jupyter notebook

# Install complete dependency stack
pip install pandas numpy matplotlib scipy pvlib geopy imageio openpyxl xlrd kgcpy

# For Windows users with permission issues (OneDrive), ensure robust Excel handling
pip install --upgrade openpyxl xlrd
```

**Quick Testing Commands:**
```bash
# Test data loading for a single site
python -c "import pandas as pd; print(pd.read_excel('Data/25_05_01_Newsites_summary.xlsx').head())"

# Verify pvlib installation
python -c "import pvlib; print(pvlib.__version__)"

# Check if all required directories exist
ls -la Data/ Results/ Code/
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
- **Site Selection**: `site_ids = ['4111846']` - Configure which sites to process
- **Directory Paths**: `data_dir`, `base_dir`, `results_dir`, `summary_dir` - File system locations
- **Analysis Period**: `num_days_to_plot = 10` - Limit processing time for large datasets
- **Plot Boundaries**: 
  - Module-level: `y_limit_module = (0,15)`, `x_limit_module = (0,60)`
  - System-level: `y_limit_inverter = (0,17)`, `x_limit_inverter = (0,1200)`
- **Physics Settings**: 
  - `use_dynamic_vth = True` - Calculate thermal voltage from actual temperature
  - `use_a_T = True` - Use ambient temperature when module temperature sensors appear faulty
- **Seasonal Mapping**: Automatic Northern/Southern hemisphere season detection based on country

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

**High-Level System Architecture:**
The analysis pipeline consists of three main components working together:
1. **Data Processing Engine**: Reconstructs I-V curves from MPP telemetry using single-diode models
2. **Statistical Analysis Engine**: Aggregates results across sites, seasons, and climates with outlier detection
3. **Theoretical Validation Engine**: LTSpice-based circuit simulation to validate empirical findings

**Data Flow Architecture:**
```
Raw Telemetry Data (Data/{site_id}/)
├── {site_id}.PAN files (module parameters)
├── {season}/optimizer_data.csv (I,V,T,P per optimizer)
├── {season}/inverter_data.csv (system-level data) 
└── {season}/site_daily_data.csv (daily aggregates)
    ↓
Generator Notebook (25_04_27_Mismatch_results_generator.ipynb)
├── I-V Curve Reconstruction Engine
│   ├── Single-diode parameter calculation (I0, IL functions)
│   ├── pvlib.pvsystem.v_from_i() integration
│   └── Series voltage summation for system curves
├── Mismatch Loss Calculator
│   ├── Sum-of-MPP calculation (individual optimizer products)
│   ├── Series-connection MPP extraction
│   └── Percentage difference: (Sum_MPP - Series_MPP)/Sum_MPP * 100
└── Timestamped Results Output
    ├── combined_data_{season}_{site_id}.xlsx
    ├── module_param_df.csv (I0, Isc, Voc, FF, Pmp per optimizer)
    ├── Animated GIFs showing I-V evolution
    └── Time-series plots and statistical summaries
    ↓
Results/v_from_i_combined/{site_id}_{season}_{timestamp}/
    ↓
Analyzer Notebook (25_07_22_Mismatch_results_analyser.ipynb)
├── Cross-Site Statistical Analysis
│   ├── Climate zone integration (Köppen-Geiger via kgcpy)
│   ├── Geographic correlation analysis (latitude, longitude)
│   └── System configuration impact (orientation, shading, size)
├── Bypass Diode Detection Engine
│   ├── Outlier detection: IQR-based Voc/Isc filtering
│   ├── Diode activation patterns: Voc loss without Isc change
│   └── "no_diode" filtered datasets for true mismatch analysis
├── Section 6: Daily Analysis Subsystem
│   ├── Daily-level granularity processing
│   ├── Vectorized operations for performance
│   └── Excel output: daily_mismatch_summary.xlsx (20 columns)
└── Comparative Visualizations
    ├── Before/after diode filtering comparisons
    ├── Seasonal and climate zone correlations
    └── System parameter impact analysis

Parallel Validation Track:
LTSpice Simulation Data (25_04_04_bypass.xlsx)
├── 4,851 I-V data points (normal/1-diode/2-diode scenarios)
├── 0-48.5V range with Bomen Solar Farm specifications
└── 72-cell module circuit models with bypass diodes
    ↓
PV I-V Analysis (25_07_22_PV_IV_Curve_Analysis.ipynb)
├── Robust Excel Loading (OneDrive permission handling)
├── MPP Detection and Power Loss Quantification
├── Publication-Quality Plotting Parameters
└── Theoretical Validation: 34.7% (1 diode), 41.8% (2 diodes)
    ↓
Integration with Empirical Analysis
└── Cross-validation of bypass diode power loss predictions
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

## Core Analysis Logic and Functions

**Critical Function Architecture:**

**Single-Diode Model Parameter Extraction (`Code/25_04_27_Mismatch_results_generator.ipynb`):**
```python
def I0(I, V, Rs, Rsh, n, N, vth):
    """Calculate dark saturation current from MPP data"""
    # Implements: I0 = [I*(1+Rs/Rsh) - V/Rsh] / [1 - I*Rs/V] * (n*N*vth/V) * exp(-(V+I*Rs)/(n*N*vth))

def IL(I, V, Rs, Rsh, n, N, vth, I0):
    """Calculate light-generated current from MPP data and I0"""
    # Implements: IL = I*(1+Rs/Rsh) + V/Rsh + I0*(exp((V+I*Rs)/(n*N*vth)) - 1)
```

**I-V Curve Reconstruction Pipeline:**
1. **Parameter Extraction**: Extract Rs, Rsh, n, Ncells from .PAN files using text parsing
2. **Thermal Voltage Calculation**: `vth = k*T/q` (dynamic) or constant at 25°C
3. **Diode Parameter Calculation**: Use measured MPP (I,V) to solve for I0 and IL
4. **Full Curve Generation**: `pvlib.pvsystem.v_from_i(current_array, IL, I0, Rs, Rsh, n*N*vth)`
5. **Series Combination**: Sum voltages at constant current across all optimizers

**Mismatch Loss Calculation Engine:**
The core research contribution - quantifying power losses from module-level mismatches:

```python
# Two power calculations compared:
sum_of_mpp = sum(Vi * Ii for all optimizers i)  # Independent operation
series_mpp = max(Vseries * I) where Vseries = sum(Vi(I))  # Series constraint

# Mismatch loss percentage:
mismatch_loss = (sum_of_mpp - series_mmp) / sum_of_mpp * 100
```

**Key Insight**: Series-connected modules operate at the same current, forcing higher-performing modules to operate below their individual MPP, creating mismatch losses.

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

## Troubleshooting and Development Guide

**Common Data Issues and Solutions:**

**1. Missing/Corrupted .PAN Files:**
```python
# Check required parameters exist:
required_params = ['RSerie', 'RShunt', 'NCelS', 'Gamma']
# PAN files use text parsing - ensure exact spelling matches
```

**2. Timestamp Synchronization Problems:**
```python
# Multiple formats supported automatically:
timestamp_formats = ["%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M"]
# Data synchronized to 5-minute intervals across all optimizers
```

**3. Zero Power Data Filtering:**
```python
# Automatically filters timestamps where ALL optimizers report zero power
all_power_zero = all(power_optimizer_i == 0 for all optimizers)
if all_power_zero: continue  # Skip timestep
```

**4. Temperature Sensor Issues:**
```python
# Use ambient temperature when module sensors appear faulty
use_a_T = True  # Replaces panel_temperature with temperature
# Check for unrealistic module temperatures (>80°C in normal conditions)
```

**5. Excel Permission Issues (OneDrive):**
```python
# LTSpice notebook uses temporary file method:
import tempfile, shutil
temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
shutil.copy2(original_path, temp_file.name)
df = pd.read_excel(temp_file.name)
```

**6. Memory Issues with Large Datasets:**
- Reduce `num_days_to_plot` from 10 to 3-5 days
- Process sites individually rather than in batch
- Use vectorized operations in daily analysis (Section 6)

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

## Statistical Analysis and Validation Framework

**Bypass Diode Detection Algorithm (`25_07_22_Mismatch_results_analyser.ipynb`):**
```python
# IQR-based outlier detection for diode activation:
def detect_diode_activation(voc_data, isc_data):
    # Voc outliers: IQR factor 1.5 for high values, percentile-based for losses
    voc_q1, voc_q3 = np.percentile(voc_data, [25, 75])
    voc_iqr = voc_q3 - voc_q1
    voc_outliers = (voc_data > voc_q3 + 1.5*voc_iqr) | (voc_data < voc_q1 - 1.5*voc_iqr)
    
    # Isc outliers: IQR factor 1.5 (low) and 3.0 (high) with minimum IQR enforcement  
    isc_q1, isc_q3 = np.percentile(isc_data, [25, 75])
    isc_iqr = max(isc_q3 - isc_q1, min_iqr_threshold)
    isc_outliers = (isc_data < isc_q1 - 1.5*isc_iqr) | (isc_data > isc_q3 + 3.0*isc_iqr)
    
    # Diode activation: Voc loss without Isc change, or Voc high + Isc low
    return voc_outliers | isc_outliers
```

**Climate Zone Integration:**
```python
# Köppen-Geiger climate classification using kgcpy library:
from geopy.geocoders import Nominatim
from kgcpy import lookupCZ

def get_climate_zone(address):
    geolocator = Nominatim(user_agent="solar_analysis")
    location = geolocator.geocode(address)
    if location:
        return lookupCZ(location.latitude, location.longitude)
    return None
```

**Cross-Site Statistical Validation:**
- **Physical Constraint**: `sum_of_mpp >= series_mpp` always (verified computationally)
- **Diurnal Pattern Validation**: FFT analysis confirms expected solar irradiance cycles
- **Geographic Correlation**: Latitude vs. mismatch loss correlation analysis
- **Climate Zone Impact**: Köppen-Geiger classification impact on seasonal variations
- **System Configuration**: Shading, orientation, and size impact quantification

**Theoretical Validation via LTSpice:**
- **34.7% power loss** with 1 bypass diode activated (validated against 4,851 simulation points)
- **41.8% power loss** with 2 bypass diodes activated 
- **Progressive degradation patterns** match empirical SolarEdge observations
- **Voltage shift analysis** confirms bypass diode activation signatures

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