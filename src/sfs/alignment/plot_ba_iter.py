import os.path
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pdir = "/home/sberton2/Downloads/MSP/tile_0/ba_iter*/"

# Read the file and extract data
match_offset_stats_path = glob(f"{pdir}run-mapproj_match_offset_stats.txt")
match_offset_stats_path = sorted(match_offset_stats_path, key=lambda name: int(name.split('ba_iter')[-1].split('/run')[0]))

def plot_camera_residuals(residual_paths, column='median', title=''):

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for idx, residual_file in enumerate(residual_paths):
        residuals_df = pd.read_csv(residual_file, sep=', ', skiprows=1)
        print(residuals_df.columns)
        residuals_df['# Image'] = residuals_df['# Image'].apply(os.path.basename)
        print(residuals_df)
        residuals_df[column].hist(bins=50, label=idx, ax=ax)

    if title is not None:
        plt.title(title)
    plt.legend()
    plt.show()

initial_residuals_path = glob(f"{pdir}run-initial_residuals_stats.txt")
initial_residuals_path = sorted(initial_residuals_path, key=lambda name: int(name.split('ba_iter')[-1].split('/run')[0]))
plot_camera_residuals(initial_residuals_path, column='median', title='Initial Residuals')

final_residuals_path = glob(f"{pdir}run-final_residuals_stats.txt")
final_residuals_path = sorted(final_residuals_path, key=lambda name: int(name.split('ba_iter')[-1].split('/run')[0]))
plot_camera_residuals(final_residuals_path, column='median', title='Final Residuals')

def match_offset_stats_to_df(file_path):
    # Initialize lists to store the data
    image_names = []
    percentiles_25 = []
    percentiles_50 = []
    percentiles_75 = []
    percentiles_85 = []
    percentiles_95 = []
    counts = []

    # Read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Stop reading when reaching the specified row
            if "Percentiles of distances between matching pixels after mapprojecting onto DEM." in line:
                break
            # Skip comment lines
            if line.startswith("#"):
                continue
            # Split the line into parts
            parts = line.split()
            image_name = parts[0].rsplit('.', 1)[0]  # Remove the extension
            image_names.append(image_name)
            percentiles_25.append(float(parts[1]))
            percentiles_50.append(float(parts[2]))
            percentiles_75.append(float(parts[3]))
            percentiles_85.append(float(parts[4]))
            percentiles_95.append(float(parts[5]))
            counts.append(int(parts[6]))

    # Create a DataFrame
    data = {
        'image_name': image_names,
        '25%': percentiles_25,
        '50%': percentiles_50,
        '75%': percentiles_75,
        '85%': percentiles_85,
        '95%': percentiles_95,
        'count': counts
    }
    return pd.DataFrame(data)

dfs = {}
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
for iter in match_offset_stats_path:

    iter_name = iter.split('ba_iter')[-1].split('/run')[0]
    dfs[iter_name] = match_offset_stats_to_df(iter)

    print(iter_name, len(dfs[iter_name]), dfs[iter_name].median(axis=0, numeric_only=True))
    dfs[iter_name]['85%'].hist(bins=50, label=iter_name, ax=ax)
plt.legend()
plt.show()

def plot_percentiles(df, title=None):
    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot the percentiles as stacked bars
    percentile_columns = ['25%', '50%', '75%', '85%', '95%']
    bottom = np.zeros(len(df))
    for col in percentile_columns:
        ax1.bar(df['image_name'], df[col], bottom=bottom, label=col, alpha=0.7)
        bottom += df[col]

    # Add a secondary y-axis for the count data as a curve
    # ax2 = ax1.twinx()
    # ax2.plot(df['image_name'], df['count'], color='red', marker='o', linestyle='-', linewidth=2, label='Count')
    # ax2.set_ylabel('Count')

    # Set labels and title
    ax1.set_xlabel('Image Name')
    ax1.set_ylabel('Percentile Residuals')
    ax1.set_title('Comparison of Percentile Residuals and Data Points for Each Image')

    # Combine legends from both y-axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    # lines_2, labels_2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    if title is not None:
        plt.title(title)

    # Show the plot
    plt.semilogy()
    plt.tight_layout()
    plt.show()

for iter, df in dfs.items():
    plot_percentiles(df, title=f'Percentiles at iter #{iter}')
