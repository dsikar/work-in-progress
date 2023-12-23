import pickle
import sys
import os
import matplotlib.pyplot as plt

# define the perturbation type and label
func_name = 'gaussian_noise'
func_label = 'Gaussian Noise'

# Define a formatter function
def custom_formatter(x):
    # name of perturbation in PERTURBATION_LEVELS
    if x.name == func_name:
        return ['%.2f' % item for item in x]
    else:
        return ['%.4f' % item for item in x]

# get the parent directory
parent_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = f'{parent_dir}/data'
pkl_file_path = f"{data_dir}/vanilla_cnn_mnist_20231223100837_{func_name}_levels.pkl"

with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)

x_values = []
y_values = []

for i in range(0, len(data['results'])):
    print(data['results'][i]['accuracy'])
    print(i+1)
    x_values.append(i+1)
    y_values.append(data['results'][i]['accuracy'])

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, marker='o')
plt.title(f'Accuracy % vs {func_label} Level')
plt.xlabel(f'{func_label} Level')
plt.ylabel('Accuracy %')
plt.grid(True)
plt.savefig(f'{func_name}_vs_accuracy.png')
plt.show()

# Python
import pandas as pd

df = pd.DataFrame(data['results'])
df[func_name] = x_values
df.to_csv(f'{data_dir}/{func_name}_results.csv', index=False)

# Reorder the columns
df = df[[func_name, 'accuracy', 'bd', 'kl','hi']]

# Convert to LaTeX and print
latex_table = df.apply(custom_formatter).to_latex(index=False)
print(latex_table)

with open(f'{data_dir}/{func_name}_latex_table.tex', 'w') as f:
    f.write(latex_table)



