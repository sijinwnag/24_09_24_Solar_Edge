
# LLM Instruction: Daily Mismatch Loss Analysis

## 1. Objective

Write a Python script that analyzes solar energy mismatch loss on a **daily** basis, providing more granular insights than the previous seasonal analysis. The script should process raw data, calculate daily metrics, merge with site metadata, and generate visualizations.

## 2. Input Data & Configuration

-   **Results Folder:** The primary input directory containing the raw data.
    -   Path: `C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\24_09_24_Solar_Edge\Results\v_from_i_combined\25_06_24_Results`
-   **Site Summary File:** An Excel file with metadata for each site.
    -   Path: `C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\24_09_24_Solar_Edge\Data\25_05_01_Newsites_summary.xlsx`
-   **Old Results (for comparison/merging):**
    -   Path: `C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\24_09_24_Solar_Edge\Results\25_07_02_Old_Sites\25_07_01_old_summary.xlsx`
-   **Output Folder:** The script should save all generated files (Excel reports, plots) into a new folder to keep the results organized.
    -   Path: `C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\24_09_24_Solar_Edge\Results\daily_analysis_results`
    -   *The script should create this directory if it doesn't exist.*

## 3. Core Logic: Daily Sampling and Analysis

The main goal is to shift from seasonal to daily analysis. Here's how to implement it:

1.  **Initialization:**
    -   Import necessary libraries: `pandas`, `numpy`, `os`, `matplotlib.pyplot`, `geopy`, `kgcpy`.
    -   Define all input and output paths as variables.
    -   Create an empty DataFrame, `daily_results_df`, to store the results of the daily analysis. The columns should be: `Site ID`, `Season`, `Date`, `Mean Period (h)`, `Mismatch Loss (%)`.

2.  **Data Processing Loop:**
    -   Iterate through each subfolder in the `ResultsFolder`. Each folder represents a specific `site_id` and `season`.
    -   Inside each folder, find the relevant data file. The filename **must contain both `combined_data` and `no_diode`** and end with `.csv` or `.xlsx`.
    -   Load this file into a pandas DataFrame.

3.  **Daily Resampling and Calculation:**
    -   Convert the `Timestamp` column to datetime objects.
    -   Group the DataFrame by **date**.
    -   For each day's data group:
        -   Perform the same calculations as in the original notebook's seasonal analysis:
            -   **FFT Analysis:** Calculate the centroid frequency and convert it to the mean period in hours (`T_centroid_h`).
            -   **Mismatch Loss:** Calculate the mismatch loss in percentage. `(E_mpp - E_series) / E_mpp * 100`.
        -   Append a new row to `daily_results_df` with the `Site ID`, `Season`, the specific `Date` of the sample, and the calculated `Mean Period (h)` and `Mismatch Loss (%)`.

4.  **Data Enrichment:**
    -   After processing all files, merge `daily_results_df` with the `site_summary_df` (from the site summary Excel file) based on `Site ID`.
    -   Add the climate zone information for each site using the `geopy` and `kgcpy` libraries, similar to the original notebook.

5.  **Output Generation:**
    -   Save the final, enriched DataFrame to an Excel file named `daily_mismatch_summary.xlsx` in the `daily_analysis_results` output folder.

## 4. Visualization

Generate the following plots based on the **daily** data. Save each plot as a high-resolution PNG file in the output folder.

1.  **Overall Mismatch Loss Distribution:**
    -   A histogram of the `Mismatch Loss (%)` column from the daily results.
    -   Title the plot with the mean and standard deviation of the mismatch loss.

2.  **Mismatch Loss vs. Mean Period:**
    -   A scatter plot of `Mismatch Loss (%)` vs. `Mean Period (h)`.
    -   Create two versions of this plot:
        -   One colored by `Orientation` (Single/Multi).
        -   One colored by `Shade` (Yes/No).
    -   Include a linear regression fit (a straight line) and display the slope for each category (e.g., for 'Single' and 'Multi' orientations separately).

3.  **Grouped Bar Charts:**
    -   **By Climate Zone and Shading:** A bar chart showing the mean mismatch loss for each climate zone, with bars grouped by whether the site is shaded or not. Include error bars representing the standard deviation.
    -   **By Season:** A bar chart showing the mean mismatch loss for each season (Summer, Autumn, Winter, Spring). Include error bars.
    -   **By Orientation:** A bar chart showing the mean mismatch loss for each orientation type. Include error bars.

## 5. Code Structure and Best Practices

-   Organize the code into functions for clarity (e.g., a function for data processing, a function for plotting).
-   Use meaningful variable names.
-   Add comments to explain complex parts of the code.
-   Ensure all file paths are handled correctly, especially when creating the output directory and saving files.
-   The script should be executable from the command line.
