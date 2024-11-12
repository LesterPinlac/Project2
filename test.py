import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim.lr_scheduler import StepLR  # Import the scheduler

# Define the device first
device = torch.device("cpu")  # Change to "cuda" if a GPU is available

# Define the input image shape
input_shape = (500, 500, 3)

# Establish the train and validation data directory
train_dir = './train'
val_dir = './valid'
test_dir = './test'

# Data augmentation for the training data
train_transforms = transforms.Compose([
    transforms.Resize((input_shape[0], input_shape[1])),
    transforms.RandomResizedCrop(input_shape[0]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
])

# Only re-scaling for the validation data
val_transforms = transforms.Compose([
    transforms.Resize((input_shape[0], input_shape[1])),
    transforms.ToTensor(),
])

# Create the train and validation datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

# Create the train and validation dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Increased batch size
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Print dataset sizes
print(f'Train dataset size: {len(train_dataset)}')
print(f'Validation dataset size: {len(val_dataset)}')

# Define the neural network architecture with an additional convolutional layer and reduced dropout
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)  # New conv layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(256 * 31 * 31, 512)  # Adjusted based on new layer
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 3)
        self.dropout = nn.Dropout(0.3)  # Reduced dropout

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))  # Forward pass through new layer
        x = x.view(-1, 256 * 31 * 31)  # Adjusted flatten dimensions
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Instantiate the model
model = CustomCNN()

# Move the model to the device
model.to(device)

# Print the model architecture
print(model)
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # Reduced learning rate
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # Learning rate scheduler

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Ensure data is on the correct device
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct / total}%')

    # Step the learning rate scheduler
    scheduler.step()