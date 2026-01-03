# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# define the data directroy:
string_power_dir = r"C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\24_09_24_Solar_Edge\Results\v_from_i_combined\25_08_25_Results\4111492_December_20250825_121529\pmppt_data.xlsx"
mod_power_dir = r"C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\24_09_24_Solar_Edge\Results\v_from_i_combined\25_08_25_Results\4111492_December_20250825_121529\iv_sum_data.xlsx"

# read the data
string_power_df = pd.read_excel(string_power_dir)
mod_power_df = pd.read_excel(mod_power_dir)

# define the font size
title_size = 22
legend_size = 18
axis_size = 20

# # print the columns to verify
# print(string_power_df.columns)
# print(mod_power_df.columns)

# ensure the Timestamp column is in datetime format
string_power_df['Timestamp'] = pd.to_datetime(string_power_df['Timestamp'])
mod_power_df['Timestamp'] = pd.to_datetime(mod_power_df['Timestamp'])

# merge the two dataframes on Timestamp
merged_df = pd.merge(string_power_df, mod_power_df, on='Timestamp', suffixes=('_string', '_mod'))

# print the columns to verify after merge
print(merged_df.columns)

# crate a subplot: top is the power comparison, bottom is the difference
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
# plot the power comparison
ax1.plot(merged_df['Timestamp'], merged_df['Pmppt (W)'], label='Series connected string power', color='blue')
ax1.plot(merged_df['Timestamp'], merged_df['Sum of I*V (W)'], label='Available module power', color='orange')
ax1.set_ylabel('Power (W)', fontsize=axis_size)
ax1.set_title('Power Comparison: String vs Module Sum', fontsize=title_size)
ax1.legend(fontsize=legend_size)
# ax1.grid()
# plot the difference
ax2.plot(merged_df['Timestamp'], - merged_df['Pmppt (W)'] + merged_df['Sum of I*V (W)'], label='Power Difference (W)', color='green')
ax2.set_ylabel('Difference (W)', fontsize=axis_size)
# ax2.set_title('Power Difference Over Time')
ax2.set_xlabel('Timestamp', fontsize=axis_size)
ax2.legend(fontsize=legend_size)
# ax2.grid()
# only show month and date in xticks
ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))
# adjust layout
plt.tight_layout()
# update the x and y tick size
ax1.tick_params(axis='both', which='major', labelsize=axis_size)
ax2.tick_params(axis='both', which='major', labelsize=axis_size)
# make the x tick appear every 2 days
ax2.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=2))
# show the plots
plt.show()