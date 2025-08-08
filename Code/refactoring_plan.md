# Refactoring Plan: `25_04_27_Mismatch_results_generator.ipynb`

## 1. Objective

The primary objective is to refactor the Jupyter notebook `25_04_27_Mismatch_results_generator.ipynb` into a more robust, maintainable, and reusable object-oriented structure. This will be achieved by creating a dedicated Python class to encapsulate the entire data analysis workflow and a new, simplified Jupyter notebook that utilizes this class.

## 2. Deliverables

1.  **`mismatch_analysis.py`**: A Python script file containing the `MismatchAnalysis` class.
2.  **`25_08_04_Mismatch_results_generator_refactored.ipynb`**: A new Jupyter notebook that serves as a high-level interface for running the analysis using the `MismatchAnalysis` class.

## 3. `mismatch_analysis.py`: Class `MismatchAnalysis` Specification

This class will contain all the logic for the mismatch analysis.

### 3.1. Class Attributes

-   `site_id` (str): The ID of the site to be analyzed.
-   `season` (str): The season for the analysis (e.g., 'spring', 'summer').
-   `data_dir` (str): The root directory containing the raw data.
-   `results_dir` (str): The root directory where results will be saved.
-   `summary_dir` (str): The path to the site summary Excel file.
-   `summary_df` (pd.DataFrame): A DataFrame holding the data from the summary file.
-   `merged_data` (pd.DataFrame): A DataFrame containing the synchronized and merged time-series data from all optimizers.
-   `reporter_ids` (list): A list of strings, where each string is the ID of an optimizer.
-   `module_params` (dict): A dictionary to store the extracted module parameters from the `.PAN` file (e.g., series_resistance, shunt_resistance).
-   `currents` (np.array): A numpy array representing the range of currents to use for reconstructing IV curves.
-   Result DataFrames (e.g., `max_power_df_combined`, `module_param_df`): Initialized as empty pandas DataFrames to store results during the analysis.

### 3.2. Methods

#### `__init__(self, site_id, season, data_dir, results_dir, summary_dir)`

-   **Purpose:** Initializes the `MismatchAnalysis` object.
-   **Steps:**
    1.  Assign all input parameters (`site_id`, `season`, etc.) to instance attributes.
    2.  Read the Excel file from `summary_dir` into the `self.summary_df` attribute.
    3.  Initialize `self.merged_data`, `self.reporter_ids`, `self.module_params`, and all result DataFrames to their default empty states.

#### `load_and_prepare_data(self)`

-   **Purpose:** Loads, parses, cleans, and merges the raw optimizer data for the specified site and season.
-   **Steps:**
    1.  Identify the site-specific data directory using `self.site_id`.
    2.  Look up the site's country in `self.summary_df` to determine the correct hemisphere (North/South) and the corresponding month-to-season mapping.
    3.  Find the season-specific data folder within the site directory.
    4.  Scan for all `optimizer_data*.csv` files.
    5.  Handle data loading based on the number of CSV files found:
        -   **Single CSV:** If a `reporter_id` column exists, split the DataFrame into multiple DataFrames based on the unique values in this column. If not, treat the entire file as belonging to a single, default reporter.
        -   **Multiple CSVs:** Assume each CSV file corresponds to a single optimizer. Extract the `reporter_id` from each filename.
    6.  For each loaded data chunk:
        -   Rename data columns to be unique by appending the reporter ID (e.g., `panel_current` -> `panel_current_12345`).
        -   Parse the `Timestamp` column, attempting multiple common date formats for robustness.
        -   Set the `Timestamp` column as the DataFrame index.
    7.  Synchronize all individual DataFrames to a common 5-minute frequency time index. This involves finding the overlapping time window (max start time and min end time) and reindexing.
    8.  Concatenate all synchronized DataFrames into a single `self.merged_data` DataFrame.

#### `extract_module_parameters(self)`

-   **Purpose:** Reads the PV module's electrical parameters from its corresponding `.PAN` file.
-   **Steps:**
    1.  Locate the `.PAN` file within the site directory.
    2.  Read the file line by line.
    3.  Parse lines containing the keys 'RSerie', 'RShunt', 'NCelS', and 'Gamma' to extract their floating-point or integer values.
    4.  Store these values in the `self.module_params` dictionary with descriptive keys (e.g., `series_resistance`).

#### `run_analysis(self)`

-   **Purpose:** Acts as the main orchestrator for the analysis, iterating through each timestamp and calling helper methods.
-   **Steps:**
    1.  Initialize empty containers for results (DataFrames, lists for image paths for the GIF).
    2.  Loop through each row (timestamp) in `self.merged_data`.
    3.  Inside the loop, initialize containers for the current timestamp's results (e.g., a list for reconstructed curves).
    4.  For each `reporter_id` (module), fetch its voltage, current, and temperature for the current timestamp.
    5.  If the data is valid (not NaN or zero), call the helper methods in sequence:
        a.  `_calculate_i0_il`
        b.  `_reconstruct_iv_curve`
        c.  `_get_single_diode_params` (to get individual module Voc, Isc, etc.)
    6.  After iterating through all modules, call `_combine_iv_curves` to get the string-level IV curve.
    7.  Call `_extract_mpp` on the combined curve to get the string-level MPP.
    8.  Call `_update_results` to save all the calculated results from this timestamp to the main result DataFrames.
    9.  Call a plotting method to generate and save the image for the current timestamp.

#### `_calculate_i0_il(self, panel_current, panel_voltage, panel_temperature)`

-   **Purpose:** Calculates the diode saturation current (I0) and the light-generated current (IL).
-   **Inputs:** `panel_current`, `panel_voltage`, `panel_temperature` for a single module at a single point in time.
-   **Returns:** A tuple `(i0, il)`.

#### `_reconstruct_iv_curve(self, i0, il, panel_temperature)`

-   **Purpose:** Reconstructs the full IV curve for a module from its calculated parameters.
-   **Inputs:** `i0`, `il`, `panel_temperature`.
-   **Returns:** A tuple `(voltage_curve, current_curve)`.

#### `_combine_iv_curves(self, iv_curves)`

-   **Purpose:** Combines multiple module IV curves into a single series-connected string IV curve.
-   **Inputs:** A list of IV curve tuples `[(voltage_curve_1, current_curve_1), ...]`.
-   **Returns:** A tuple `(combined_voltage_curve, combined_current_curve)`.

#### `_extract_mpp(self, combined_iv_curve)`

-   **Purpose:** Finds the Maximum Power Point (MPP) of an IV curve.
-   **Inputs:** An IV curve tuple `(voltage_curve, current_curve)`.
-   **Returns:** A dictionary `{'v_mp': ..., 'i_mp': ..., 'p_mp': ...}`.

#### `_update_results(self, timestamp, module_results, combined_results, sum_of_mpp)`

-   **Purpose:** Appends the results from a single timestamp to the main result DataFrames.
-   **Inputs:** The `timestamp`, results for each module, results for the combined curve, and the sum of individual MPPs.

#### `generate_plots(self)`

-   **Purpose:** Creates and saves all summary plots after the analysis loop is complete.
-   **Steps:**
    1.  Generate and save the plot comparing `Pmppt` vs. the sum of individual module MPPs.
    2.  Generate and save the plot showing the percentage power difference over time.

#### `save_results(self)`

-   **Purpose:** Saves all tabular data and the GIF.
-   **Steps:**
    1.  Save the result DataFrames to CSV or Excel files.
    2.  Compile the saved timestamp images into a single GIF.

## 4. `25_08_04_Mismatch_results_generator_refactored.ipynb`: Notebook Workflow

The new notebook will be a simple, high-level script:

1.  **Cell 1: Imports**
    -   `from mismatch_analysis import MismatchAnalysis`
    -   `import matplotlib.pyplot as plt`
2.  **Cell 2: Configuration**
    -   Define variables for `site_id`, `season`, `data_dir`, `results_dir`, and `summary_dir`.
3.  **Cell 3: Execution**
    -   Instantiate the class: `analysis = MismatchAnalysis(...)`
    -   Call the main methods in logical order:
        ```python
        analysis.load_and_prepare_data()
        analysis.extract_module_parameters()
        analysis.run_analysis()
        analysis.generate_plots()
        analysis.save_results()
        mismatch_loss = analysis.calculate_mismatch_loss()
        ```
4.  **Cell 4: Output**
    -   Print key results, such as the final mismatch loss percentage.
    -   Use `plt.show()` to display any plots that are generated in memory.
