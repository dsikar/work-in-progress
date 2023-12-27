import pickle
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

# generic function to plot the noise level vs accuracy
# and generate the corresponding latex table
# NB func_name is the name of the perturbation in PERTURBATION_LEVELS, perturbation_levels.py

# define the perturbation type and label
#func_name = 'gaussian_noise'
#func_label = 'Gaussian Noise'

def main(func_name, func_label, pkl_file):
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
    pkl_file_path = f"{data_dir}/{pkl_file}"

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
    plt.xticks(range(11))
    plt.grid(True)
    plt.savefig(f'{data_dir}/{func_name}_vs_accuracy.png')
    plt.show()

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

if __name__ == "__main__":
    # func_name = sys.argv[1]
    # func_label = sys.argv[2]
    # pkl_file_path = sys.argv[3]
    func_name = 'frost'
    func_label = 'Frost'
    pkl_file = 'vanilla_cnn_mnist_20231227165202_frost.pkl'
    print("Running with args {}, {}, {}".format(func_name, func_label, pkl_file))
    main(func_name, func_label, pkl_file)        



