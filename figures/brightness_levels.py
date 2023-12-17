import pickle
import sys
import os

# Define a formatter function
def custom_formatter(x):
    if x.name == 'brightness_level':
        return ['%.2f' % item for item in x]
    else:
        return ['%.4f' % item for item in x]

parent_dir = "/home/daniel/git/work-in-progress/"
sys.path.append(parent_dir)
from utils.perturbation_levels_single import PERTURBATION_LEVELS
base_dir = "/home/daniel/git/work-in-progress/scripts/data"
file_path = "/home/daniel/git/work-in-progress/scripts/data/vanilla_cnn_mnist_20231216182544_brightness_0.1_to_2.0.pkl"

with open(file_path, 'rb') as file:
    data = pickle.load(file)

x_values = []
y_values = []

for i in range(0, len(data['results'])):
    print(data['results'][i]['accuracy'])
    print(PERTURBATION_LEVELS['brightness'][i]['brightness_level'])
    x_values.append(PERTURBATION_LEVELS['brightness'][i]['brightness_level'])
    y_values.append(data['results'][i]['accuracy'])

    # Python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, marker='o')
plt.title('Accuracy vs brightness Level')
plt.xlabel('brightness Level')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('brightness_Level_vs_Accuracy.png')
plt.show()

# 1. Iterate through the perturbation types
# for key in PERTURBATION_LEVELS.keys():
#     print("Perturbation type:", key)
#     # 2. Iterate through the perturbation parameters
#     for k in range(0, len(PERTURBATION_LEVELS[key])):

# Python
import pandas as pd

df = pd.DataFrame(data['results'])
df['brightness_level'] = x_values
df.to_csv(f'{base_dir}/brightness_level_results.csv', index=False)

# Reorder the columns
df = df[['brightness_level', 'accuracy', 'bd', 'kl','hi']]

# Convert to LaTeX and print
latex_table = df.apply(custom_formatter).to_latex(index=False)
print(latex_table)

with open(f'{base_dir}/brightness_levels_latex_table.tex', 'w') as f:
    f.write(latex_table)

# latex table



