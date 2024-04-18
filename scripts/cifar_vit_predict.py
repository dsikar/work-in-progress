from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import sys

# Load the datasets
train_ds, test_ds = load_dataset('cifar10', split=['train[:50000]', 'test[:10000]'])

# Define the label mappings
id2label = {id: label for id, label in enumerate(train_ds.features['label'].names)}
label2id = {label: id for id, label in id2label.items()}

# get the path
# Get the parent directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to the module search path
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

PATH = 'models/vit_cifar10' # trained on google colab, 

# prepend current_dir to PATH
PATH = os.path.join(current_dir, PATH)

# Load the saved model and image processor
model_path = PATH  # Update with the actual path to the uncompressed model directory
model = ViTForImageClassification.from_pretrained(model_path)
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

# Define the batch size and create data loaders
batch_size = 16

def process_data(examples):
    pixel_values = image_processor(examples['img'], return_tensors="pt").pixel_values
    labels = examples['label']
    return {'pixel_values': pixel_values, 'labels': labels}

train_loader = DataLoader(train_ds.with_transform(process_data), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds.with_transform(process_data), batch_size=batch_size, shuffle=False)

# Initialize the numpy arrays
train_np = np.zeros((50000, 12))
test_np = np.zeros((10000, 12))

# Evaluate on the training dataset
model.eval()
index = 0
total_rows = len(train_ds)
processed_rows = 0
correct_predictions = 0
print(f"Total rows to process: {total_rows}")
with torch.no_grad():
    for batch_index, batch in enumerate(tqdm(train_loader, desc="Training")):
        outputs = model(**batch)
        softmax_outputs = torch.softmax(outputs.logits, dim=1)
        _, predicted = torch.max(outputs.logits, 1)
        
        # Store the results in train_np
        batch_size = batch['labels'].size(0)
        train_np[index:index+batch_size, :10] = softmax_outputs.numpy()
        train_np[index:index+batch_size, 10] = batch['labels'].numpy()
        train_np[index:index+batch_size, 11] = predicted.numpy()
        
        index += batch_size
        processed_rows += batch_size
        correct_predictions += (predicted == batch['labels']).sum().item()
        
        if (batch_index + 1) % 100 == 0:
            remaining_rows = total_rows - processed_rows
            running_accuracy = correct_predictions / processed_rows
            print(f"Processed rows: {processed_rows}, Remaining rows: {remaining_rows}, Running Accuracy: {running_accuracy:.4f}")

# Evaluate on the testing dataset
index = 0
total_rows = len(test_ds)
processed_rows = 0
correct_predictions = 0
print(f"Total rows to process: {total_rows}")
with torch.no_grad():
    for batch_index, batch in enumerate(tqdm(test_loader, desc="Testing")):
        outputs = model(**batch)
        softmax_outputs = torch.softmax(outputs.logits, dim=1)
        _, predicted = torch.max(outputs.logits, 1)
        
        # Store the results in test_np
        batch_size = batch['labels'].size(0)
        test_np[index:index+batch_size, :10] = softmax_outputs.numpy()
        test_np[index:index+batch_size, 10] = batch['labels'].numpy()
        test_np[index:index+batch_size, 11] = predicted.numpy()
        
        index += batch_size
        processed_rows += batch_size
        correct_predictions += (predicted == batch['labels']).sum().item()
        
        if (batch_index + 1) % 100 == 0:
            remaining_rows = total_rows - processed_rows
            running_accuracy = correct_predictions / processed_rows
            print(f"Processed rows: {processed_rows}, Remaining rows: {remaining_rows}, Running Accuracy: {running_accuracy:.4f}")

# Save the numpy arrays and class labels
np.save('train_np.npy', train_np)
np.save('test_np.npy', test_np)
class_labels = train_ds.features['label'].names
with open('class_labels.txt', 'w') as f:
    f.write('\n'.join(class_labels))

print("Arrays and class labels saved successfully.")