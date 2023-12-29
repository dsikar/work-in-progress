import pickle
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# open the file data/vanilla_cnn_mnist_20231227192341_all.pkl
pkl_file = 'vanilla_cnn_mnist_20231227192341_all.pkl'

# get the parent directory
parent_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = f'{parent_dir}/data'
# pkl_file_path = f"{data_dir}/{pkl_file}"

# with open(pkl_file_path, 'rb') as file:
#     data = pickle.load(file)

# # put data['results'] in dataframe
# results_df = pd.DataFrame(data['results'])

# # save results_df to csv
# results_df.to_csv(f'{data_dir}/vanilla_cnn_mnist_20231227192341_all.csv', index=False)

# Loading the CSV file
file_path = f'{data_dir}/vanilla_cnn_mnist_20231227192341_all.csv'
data = pd.read_csv(file_path)

# # Displaying the first few rows of the file for an overview
# data.head()

# # Filtering the data for the case of "brightness" noise type and "hi" distance metric
# brightness_data = data[data['noise_type'] == 'brightness'][['index', 'hi', 'accuracy']]

# # Plotting the relationship between "hi" distance and accuracy for "brightness" noise type
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=brightness_data, x='hi', y='accuracy')
# plt.title('Correlation of HI Distance with Accuracy for Brightness Noise')
# plt.xlabel('HI Distance')
# plt.ylabel('Accuracy (%)')
# plt.grid(True)
# plt.show()

# Calculating the correlation coefficients between distance metrics and accuracy for each noise type

# Getting the unique noise types
noise_types = data['noise_type'].unique()

# Creating an empty dataframe for the correlation coefficients
correlation_df = pd.DataFrame(index=['bd', 'kl', 'hi'], columns=noise_types)

# Calculating the correlation coefficients
for noise in noise_types:
    for metric in ['bd', 'kl', 'hi']:
        corr = data[data['noise_type'] == noise][metric].corr(data[data['noise_type'] == noise]['accuracy'])
        correlation_df.loc[metric, noise] = corr

correlation_df
# format all values to 2 decimal places
correlation_df = correlation_df.applymap('{:.2f}'.format)

# save as tex file
correlation_df.to_latex(f'{data_dir}/vanilla_cnn_mnist_20231227192341_all_correlations.tex')

# bar chart

# Preparing the data for the bar chart with one set for each noise type
correlation_df_transposed = correlation_df.T.reset_index()
correlation_df_melted = correlation_df_transposed.melt(id_vars='index', var_name='Distance Metric', value_name='Correlation Coefficient')

# Converting 'Correlation Coefficient' to numeric for plotting
correlation_df_melted['Correlation Coefficient'] = pd.to_numeric(correlation_df_melted['Correlation Coefficient'])

# Plotting the bar chart with one set for each noise type
plt.figure(figsize=(15, 8))
bar_plot = sns.barplot(x='index', y='Correlation Coefficient', hue='Distance Metric', data=correlation_df_melted)
plt.title('Correlation Coefficients of Distance Metrics with Accuracy for Each Noise Type')
plt.xlabel('Noise Type')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45)
plt.legend(title='Distance Metric')
plt.grid(axis='y')
plt.show()

# save as png
bar_plot.figure.savefig(f'{data_dir}/vanilla_cnn_mnist_20231227192341_all_correlations.png')

##########################
# plot HI vs Noise Level #
##########################

# Adjusting the 'index' (noise level) by 1 and rounding 'hi' to two decimal places
data['index_adjusted'] = data['index'] + 1
data['hi_rounded'] = data['hi'].round(2)

# Re-importing numpy and using plt.cm.viridis for the plot
plt.figure(figsize=(15, 8))

# Getting the viridis color map
colors = plt.cm.viridis(np.linspace(0, 1, len(data['noise_type'].unique())))

# Looping through each noise type to plot with colors from viridis
for i, noise_type in enumerate(data['noise_type'].unique()):
    subset = data[data['noise_type'] == noise_type]
    sns.lineplot(data=subset, x='index_adjusted', y='hi_rounded', label=noise_type, color=colors[i])

plt.title('HI Distance vs Noise Level (Adjusted) for All Noise Types')
plt.xlabel('Noise Level (Adjusted)')
plt.ylabel('HI Distance (Rounded to Two Decimal Places)')
plt.legend(title='Noise Type')
plt.grid(True)
plt.show()

# save as png
plt.savefig(f'{data_dir}/vanilla_cnn_mnist_20231227192341_all_HI_vs_noise_level.png')

##########################
# plot BD vs Noise Level #
##########################

# Adjusting the 'bd' data to two decimal places for the plot
data['bd_rounded'] = data['bd'].round(2)

# Plotting BD vs adjusted level for all noise types using the Viridis color map
plt.figure(figsize=(15, 8))

# Looping through each noise type to plot
for i, noise_type in enumerate(data['noise_type'].unique()):
    subset = data[data['noise_type'] == noise_type]
    sns.lineplot(data=subset, x='index_adjusted', y='bd_rounded', label=noise_type, color=colors[i])

plt.title('Bhattacharyya Distance vs Noise Level for All Noise Types')
plt.xlabel('Noise Level (Adjusted)')
plt.ylabel('Bhattacharyya Distance (Rounded to Two Decimal Places)')
plt.legend(title='Noise Type')
plt.grid(True)
plt.show()

# save as png
plt.savefig(f'{data_dir}/vanilla_cnn_mnist_20231227192341_all_BD_vs_noise_level.png')

##########################
# plot KL vs Noise Level #
##########################

# Adjusting the 'kl' data to two decimal places for the plot
data['kl_rounded'] = data['kl'].round(2)

# Plotting KL vs adjusted level for all noise types using the Viridis color map
plt.figure(figsize=(15, 8))

# Looping through each noise type to plot
for i, noise_type in enumerate(data['noise_type'].unique()):
    subset = data[data['noise_type'] == noise_type]
    sns.lineplot(data=subset, x='index_adjusted', y='kl_rounded', label=noise_type, color=colors[i])

plt.title('KL Divergence vs Noise Level (Adjusted) for All Noise Types')
plt.xlabel('Noise Level (Adjusted)')
plt.ylabel('KL Divergence (Rounded to Two Decimal Places)')
plt.legend(title='Noise Type')
plt.grid(True)
plt.show()

# save as png
plt.savefig(f'{data_dir}/vanilla_cnn_mnist_20231227192341_all_KL_vs_noise_level.png')

###############################
# SIDE BY SIDE
###############################

# Creating subplots for HI, BD, and KL
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

# Plotting HI
for i, noise_type in enumerate(data['noise_type'].unique()):
    subset = data[data['noise_type'] == noise_type]
    sns.lineplot(ax=axes[0], data=subset, x='index_adjusted', y='hi_rounded', label=noise_type, color=colors[i])
axes[0].set_title('Histogram Intersection vs Noise Level')
axes[0].set_xlabel('Noise Level')
axes[0].set_ylabel('Histogram Intersection')
axes[0].legend().remove()
axes[0].grid(True)

# Plotting BD
for i, noise_type in enumerate(data['noise_type'].unique()):
    subset = data[data['noise_type'] == noise_type]
    sns.lineplot(ax=axes[1], data=subset, x='index_adjusted', y='bd_rounded', label=noise_type, color=colors[i])
axes[1].set_title('Bhattacharyya Distance vs Noise Level')
axes[1].set_xlabel('Noise Level')
axes[1].set_ylabel('Bhattacharyya Distance Distance')
axes[1].legend().remove()
axes[1].grid(True)

# Plotting KL
for i, noise_type in enumerate(data['noise_type'].unique()):
    subset = data[data['noise_type'] == noise_type]
    sns.lineplot(ax=axes[2], data=subset, x='index_adjusted', y='kl_rounded', label=noise_type, color=colors[i])
axes[2].set_title('KL Divergence vs Noise Level')
axes[2].set_xlabel('Noise Level')
axes[2].set_ylabel('KL Divergence')
axes[2].legend(title='Noise Type', loc='upper right')
axes[2].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()

# save as png
plt.savefig(f'{data_dir}/vanilla_cnn_mnist_20231227192341_all_HI_BD_KL_vs_noise_level.png')

#########################
# Accuracy vs Distance
#########################

# Creating subplots for Accuracy vs HI, BD, and KL
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

# Plotting Accuracy vs HI
for i, noise_type in enumerate(data['noise_type'].unique()):
    subset = data[data['noise_type'] == noise_type]
    sns.scatterplot(ax=axes[0], data=subset, x='hi_rounded', y='accuracy', label=noise_type, color=colors[i])
axes[0].set_title('Accuracy vs HI Distance')
axes[0].set_xlabel('HI Distance (Rounded)')
axes[0].set_ylabel('Accuracy')
axes[0].legend(title='Noise Type', loc='upper right')

# Plotting Accuracy vs BD
for i, noise_type in enumerate(data['noise_type'].unique()):
    subset = data[data['noise_type'] == noise_type]
    sns.scatterplot(ax=axes[1], data=subset, x='bd_rounded', y='accuracy', label=noise_type, color=colors[i])
axes[1].set_title('Accuracy vs BD Distance')
axes[1].set_xlabel('BD Distance (Rounded)')
axes[1].set_ylabel('Accuracy')
axes[1].legend().remove()

# Plotting Accuracy vs KL
for i, noise_type in enumerate(data['noise_type'].unique()):
    subset = data[data['noise_type'] == noise_type]
    sns.scatterplot(ax=axes[2], data=subset, x='kl_rounded', y='accuracy', label=noise_type, color=colors[i])
axes[2].set_title('Accuracy vs KL Divergence')
axes[2].set_xlabel('KL Divergence (Rounded)')
axes[2].set_ylabel('Accuracy')
axes[2].legend().remove()

# Adjust layout
plt.tight_layout()
plt.show()
