# DONE - Chris M. added lecun. domain to whitelist
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime

# vanilla CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Function to calculate and print shape and number of parameters
def print_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        param_count = torch.numel(param)
        total_params += param_count
        print(f"{name} - shape: {param.shape}, parameters: {param_count}")
    print(f"Total number of parameters: {total_params}")

# Define transform to normalize data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download, if required, and load the training data
trainset = datasets.MNIST('data/', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download, if required, and load the test data
testset = datasets.MNIST('data/', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Create a network instance
net = Net()

# Print the shape and number of parameters for each layer
print_parameters(net)

# Train the model
num_epochs = 10
optimizer = optim.SGD(net.parameters(), lr=0.01) 
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    running_loss = 0.0
    running_corrects = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99: # Print every 100 mini-batches
            epoch_loss = running_loss / 100
            epoch_acc = running_corrects.double() / ((i + 1) * 64)
            print('[Epoch %d, Batch %5d] Loss: %.3f Acc: %.3f' % (epoch + 1, i + 1, epoch_loss, epoch_acc))
            running_loss = 0.0
            running_corrects = 0.0

print('Finished Training')

# Save the trained model weights to a file
current_datetime = datetime.datetime.now()

# Format the date and time string
date_time_string = current_datetime.strftime("%Y%m%d%H%M")

PATH = 'models/mnist_vanilla_cnn_local_'+ date_time_string + '.pth'
torch.save(net.state_dict(), PATH)
print('Weights saved to ' + PATH)