# Multi-Orientation Solar Array Mismatch Analysis

## Overview

This module (`25_09_05_Mismatch_results_generator_multi.py`) implements an enhanced version of the original mismatch analysis workflow specifically designed for multi-orientation solar installations. Instead of treating all modules as a single series string, it groups modules by panel current percentiles at each timestamp to better represent the multiple orientation configurations.

## Key Features

### 1. Current-Based Module Grouping
- **Per-timestamp Analysis**: At each 5-minute interval, modules are grouped based on their panel current percentiles
- **Intelligent Grouping**: Uses `pandas.qcut()` with fallback to rank-based slicing for flat distributions
- **Edge Case Handling**: Manages insufficient modules, duplicate current values, and invalid data

### 2. Multi-String Power Calculation
- **Group-wise Series Calculation**: Applies traditional pvlib series combination within each current group
- **Power Summation**: Combines power from all groups: `P_series_multi = Σ P_string_g`
- **Comparison Metrics**: Calculates both traditional single-string and multi-string mismatch losses

### 3. Enhanced Visualizations
- **Color-coded Plots**: Modules colored by current group assignment in I-V plots
- **Group Legends**: Shows current range for each orientation group
- **Comparative Analysis**: Displays both traditional and multi-string power calculations

## Multi-Orientation Site Configuration

The following sites are configured for multi-orientation analysis:

```python
MULTI_ORIENTATION_SITES = {
    '3455043': 3,  # 3 orientations
    '4111492': 4,  # 4 orientations  
    '4111800': 4,  # 4 orientations
    '4118327': 4,  # 4 orientations
    '3794347': 6,  # 6 orientations
    '4173851': 4   # 4 orientations
}
```

## Usage

### Basic Usage
```python
from 25_09_05_Mismatch_results_generator_multi import run_multi_orientation_analysis

# Process all multi-orientation sites for spring season
run_multi_orientation_analysis(
    site_ids=None,  # All multi-orientation sites
    seasons=['spring'],
    num_days_to_plot=5
)
```

### Custom Analysis
```python
# Process specific sites and seasons
run_multi_orientation_analysis(
    site_ids=['3455043', '4111492'],  # Specific sites
    seasons=['spring', 'summer'],
    num_days_to_plot=10
)
```

### Advanced Usage
```python
import importlib.util

# Load module dynamically
spec = importlib.util.spec_from_file_location('multi_analysis', '25_09_05_Mismatch_results_generator_multi.py')
multi_analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi_analysis)

# Use individual functions
panel_currents = {'mod1': 5.2, 'mod2': 7.8, 'mod3': 3.1}
reporter_ids = ['mod1', 'mod2', 'mod3']
groups, n_groups, ranges = multi_analysis.group_modules_by_current_percentiles(
    panel_currents, 3, reporter_ids
)
```

## Output Structure

The analysis generates enhanced outputs in the results directory:

```
Results/v_from_i_combined/{date_folder}/{site_id}_{month}_{timestamp}/
├── combined_data_enhanced_{season}_{site_id}.xlsx     # Enhanced metrics with multi-string analysis
├── multi_string_analysis_{site_id}.csv               # Per-timestamp group breakdown
├── long_horizontal_{timestamp}_grouped.png           # Color-coded module plots
├── combined_iv_curves_grouped.gif                   # Animated visualization
├── iv_sum_data.xlsx                                  # Sum of individual module powers
├── pmppt_data.xlsx                                   # Traditional series power
├── multi_string_data.xlsx                           # Multi-string power results
└── module_param_df.csv                              # Module parameters per timestamp
```

## Key Output Columns

### Enhanced Combined Data (`combined_data_enhanced_{season}_{site_id}.xlsx`)
- `Sum of I*V (W)`: Baseline power (sum of individual module MPP)
- `Pmppt (W)`: Traditional series-connection power
- `Multi_String_Power (W)`: Multi-string grouped power
- `Traditional_Mismatch_%`: Traditional mismatch loss percentage
- `Multi_String_Mismatch_%`: Multi-string mismatch loss percentage  
- `Improvement_%`: Improvement in percentage points

### Multi-String Analysis (`multi_string_analysis_{site_id}.csv`)
- `Timestamp`: Analysis timestamp
- `N_Groups`: Effective number of groups created
- `Group_ID`: Group identifier (1, 2, 3, ...)
- `Reporter_IDs`: Semicolon-separated list of modules in group
- `Current_Range_Min/Max`: Current range for the group
- `Group_Power`: Series power for this specific group

## Algorithm Details

### Current-Based Grouping Algorithm
1. **Extract Current Values**: Get panel current for all modules at timestamp
2. **Filter Invalid Data**: Remove zero, NaN, or negative current values  
3. **Percentile Calculation**: Use `np.linspace(0, 1, N+1)` for quantile edges
4. **Pandas qcut**: Apply `pd.qcut()` with `duplicates="drop"`
5. **Fallback Strategy**: If qcut fails, use rank-based equal-count slicing
6. **Group Assignment**: Map modules to groups (1-indexed)

### Multi-String Power Calculation
1. **Group Processing**: For each current group separately:
   - Apply single-diode I-V curve reconstruction
   - Calculate series voltage: `V_group = Σ V_module(I)`
   - Find maximum power: `P_group = max(V_group × I)`
2. **Power Summation**: `P_multi_string = Σ P_group`
3. **Comparison**: Compare with traditional `P_series = max(Σ V_module(I) × I)`

### Edge Case Handling
- **Insufficient Modules**: If fewer modules than orientations, reduce group count
- **Flat Current Distribution**: Fallback to rank-based slicing when qcut fails
- **Invalid Data**: Assign invalid modules to group 1, log warnings
- **Zero Power Timestamps**: Skip analysis when all modules report zero power

## Physical Interpretation

The multi-string approach better represents real multi-orientation systems where:
- **Different Orientations**: Modules face different directions (east/west/south)
- **Varying Irradiance**: Each orientation receives different solar irradiance
- **Current Grouping**: Modules with similar currents likely share similar conditions
- **Independent Strings**: Each orientation operates as a semi-independent string

## Performance Considerations

- **Memory Usage**: Stores detailed per-timestamp group information
- **Processing Time**: ~20% slower than single-orientation due to grouping overhead
- **Visualization**: Color-coded plots may take longer to render
- **Batch Processing**: Processes 5-day periods by default for faster execution

## Dependencies

All dependencies match the original workflow:
- `pandas`, `numpy`, `matplotlib`, `pvlib`
- `imageio`, `scipy`, `datetime`
- Standard library: `os`, `sys`, `warnings`

## Validation

The implementation includes several validation checks:
- **Physical Constraints**: `P_multi_string ≤ P_baseline` (sum of individual MPP)
- **Group Consistency**: Number of modules assigned equals reporter count
- **Current Ordering**: Groups ordered by current percentile ranges
- **Power Conservation**: All group powers sum to multi-string total

## Troubleshooting

### Common Issues
1. **Import Error**: Use `importlib.util` for dynamic import due to numeric filename
2. **No Multi-Orientation Sites**: Check `MULTI_ORIENTATION_SITES` configuration
3. **Data Loading Failures**: Verify optimizer_data.csv files exist and have required columns
4. **Memory Issues**: Reduce `num_days_to_plot` parameter for large datasets

### Debug Output
The module provides detailed console output including:
- Site processing progress
- Module parameter extraction results  
- Grouping statistics per timestamp
- Final mismatch loss summary

## Research Applications

This enhanced analysis enables investigation of:
- **Orientation Impact**: Quantify how multiple orientations affect mismatch losses
- **Seasonal Variations**: Compare traditional vs. multi-string losses across seasons
- **System Optimization**: Identify optimal grouping strategies for multi-orientation arrays
- **Performance Modeling**: Improve accuracy of multi-orientation system models