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

