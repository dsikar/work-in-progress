import pickle
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# open the file data/vanilla_cnn_mnist_20231227192341_all.pkl
pkl_file = 'vanilla_cnn_mnist_20231227192341_all.pkl'

# get the parent directory
parent_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = f'{parent_dir}/data'
pkl_file_path = f"{data_dir}/{pkl_file}"

with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)

# put data['results'] in dataframe
results_df = pd.DataFrame(data['results'])

# Extracting unique 'index' values from the results DataFrame
unique_indices = results_df['index'].unique()
unique_indices

# Creating a new DataFrame with the unique index values as "Level"
unique_indices_df = pd.DataFrame(unique_indices + 1, columns=["Level"])
unique_indices_df

# Extracting unique noise_types
unique_noise_types = results_df['noise_type'].unique()

# Creating a DataFrame with 'Level' and unique noise_types as column headers
# Start with a DataFrame that has just the 'Level' column from the unique_indices_df
noise_types_df = unique_indices_df.copy()

# Add columns for each noise type, initializing with NaNs
for noise_type in unique_noise_types:
    noise_types_df[noise_type] = float('nan')

# Set the order of the columns with 'Level' as the first column
ordered_columns = ['Level'] + list(unique_noise_types)
noise_types_df = noise_types_df[ordered_columns]

noise_types_df

# Populate the columns with the accuracy values for the corresponding noise_type and index
for index, row in results_df.iterrows():
    # The 'index' from the results_df corresponds to the 'Level' in our new DataFrame, with 1 added to it
    level = row['index'] + 1
    noise_type = row['noise_type']
    # accuracy to 2 decimal places
    accuracy = round(row['accuracy'], 2)
    # accuracy = row['accuracy']
    
    # Set the accuracy value in the correct place in the noise_types_df
    noise_types_df.loc[noise_types_df['Level'] == level, noise_type] = accuracy

noise_types_df.to_string(index=False)

# omit the first row and the first column
myplotdata = noise_types_df.iloc[1:, 1:]

# plot the data


# write noise_types_df to latex table
latex_table = noise_types_df.to_latex(index=False)

# write latex_table to file
with open(f'{data_dir}/noise_types_df_latex_table.tex', 'w') as f:
    f.write(latex_table)

# write to csv
noise_types_df.to_csv(f'{data_dir}/noise_types_df.csv', index=False)

# Load the content of the uploaded CSV file
csv_file_path = f'{data_dir}/noise_types_df.csv'

# Reading the CSV file into a DataFrame
csv_data = pd.read_csv(csv_file_path)

# Displaying the first few rows of the DataFrame to understand its structure
csv_data.head()

# plot table

# Adjusting the plot to end the x-axis at 11 and using a more diverse color palette

plt.figure(figsize=(12, 8))

# Using a color palette for better distinction
colors = plt.cm.viridis(np.linspace(0, 1, len(csv_data.columns[1:])))

for i, column in enumerate(csv_data.columns[1:]):  # Skipping the 'Level' column
    plt.plot(csv_data['Level'], csv_data[column], label=column, color=colors[i])

# Setting the x-axis limits
plt.xlim(0, 11)
# set the xticks

plt.xticks(range(1, 11))

plt.ylim(30, 100)

# add a grid
plt.grid(True)

# Labeling the axes
plt.xlabel('Noise Level')
plt.ylabel('Predictive Accuracy')

# Adding a legend
plt.legend()

# Adding a title
plt.title('Accuracy vs Noise Level, Vanilla CNN on MNIST')

# Showing the plot
plt.show()

# save the plot
plt.savefig(f'{data_dir}/accuracy_vs_noise_types_plot.png')








    

