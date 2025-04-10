import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd

# CNN definition
class InitialCNN(nn.Module):
    def __init__(self):
        super(InitialCNN, self).__init__()

        # Define convolutional layers
        # First conv layer: input channels = 3 (RGB), output channels = 32, 3x3 kernel, stride 1, padding 1
        # ReLU activation introduces non-linearity
        # MaxPool layer reduces spatial dimensions by 2 (kernel size 2)
        # Second conv layer: input channels = 32, output channels = 64, same kernel config
        # Another ReLU and MaxPool reduce dimensions again
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Define fully connected (dense) layers
        # Flatten layer converts 3D feature maps to 1D vector
        # First Linear layer: input = 64 channels * 62 * 62 (output size after 2 MaxPool layers from 250x250 input)
        # Second Linear layer outputs 2 logits for binary classification (fire, non-fire)
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 62 * 62, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

# Transfer learning extension of InitialCNN
# Transfer learning model that extends InitialCNN with an additional convolutional layer and a modified FC head
class ExtendedCNN(nn.Module):
    def __init__(self):
        super(ExtendedCNN, self).__init__()

        # Define convolutional layers identically to InitialCNN for weight compatibility
        # First conv layer: input channels = 3 (RGB), output channels = 32, 3x3 kernel, stride 1, padding 1
        # ReLU activation introduces non-linearity
        # MaxPool layer reduces spatial dimensions by 2 (kernel size 2)
        # Second conv layer: input channels = 32, output channels = 64, same kernel config
        # Another ReLU and MaxPool reduce dimensions again
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Add new convolutional layer to deepen the network
        # Third conv layer: input channels = 64, output channels = 128, 3x3 kernel, stride 1, padding 1
        # ReLU activation followed by MaxPool to reduce feature map size further
        self.additional_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Define fully connected (dense) layers
        # Flatten layer converts 3D feature maps to 1D vector
        # First Linear layer: input = 128 channels * 31 * 31 (output after third MaxPool)
        # Second Linear layer outputs 2 logits for binary classification (fire, non-fire)
        # Dropout added between FC layers to improve generalization
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 31 * 31, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.additional_conv(x)
        x = self.fc_layer(x)
        return x

# Custom dataset class that reads test images and infers class from filename prefix
class TestDatasetFromFilenames(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_paths = [os.path.join(directory, f) for f in os.listdir(directory)
                            if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = 0 if os.path.basename(image_path).lower().startswith("fire") else 1
        if self.transform:
            image = self.transform(image)
        return image, label

# Training function with accuracy and loss reporting
# Sets model to training mode, iterates through batches, computes gradients, and updates weights
# Reports average loss and accuracy per epoch

import time

def train(model, device, train_loader, criterion, optimizer):
    # Declare global lists for metrics storage
    global epoch_times, epoch_losses, epoch_accuracies

    # Set the model to training mode
    model.train()

    # Initialize list to store elapsed time for each epoch
    epoch_times = []

    # Initialize list to store loss for each epoch
    epoch_losses = []

    # Initialize list to store accuracy for each epoch
    epoch_accuracies = []

    # Loop over the number of epochs
    for epoch in range(10):
        # Capture the start time of the epoch
        start_time = time.time()

        # Initialize loss and accuracy counters for the epoch
        running_loss = 0.0
        correct, total = 0, 0

        # Iterate over batches in the training data
        for inputs, labels in train_loader:
            # Move input data and labels to the appropriate device (CPU or GPU)
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients before the backward pass
            optimizer.zero_grad()

            # Perform the forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass to compute gradients
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Accumulate the running loss
            running_loss += loss.item()

            # Get predicted class from output
            _, predicted = torch.max(outputs, 1)

            # Update total and correct prediction counters
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate average loss over the epoch
        avg_loss = running_loss / len(train_loader)

        # Calculate accuracy percentage
        accuracy = 100 * correct / total

        # Capture the end time of the epoch
        end_time = time.time()

        # Calculate elapsed time for the epoch
        elapsed_time = end_time - start_time

        # Store metrics in their respective lists
        epoch_times.append(elapsed_time)
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(accuracy)

        # Print training metrics for the epoch
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Time: {elapsed_time:.2f} seconds")

# Evaluation function to measure test accuracy
# Disables gradient tracking and compares predicted labels to true labels

def test(model, device, test_loader):
    # Set the model to evaluation mode
    model.eval()

    # Initialize counters for correct predictions and total samples
    correct, total = 0, 0

    # Disable gradient calculations for efficiency
    with torch.no_grad():
        # Iterate over the test dataset
        for inputs, labels in test_loader:
            # Move input data and labels to the appropriate device
            inputs, labels = inputs.to(device), labels.to(device)

            # Perform the forward pass
            outputs = model(inputs)

            # Get the predicted class
            _, predicted = torch.max(outputs, 1)

            # Update counters
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print the final test accuracy
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Main function for training and evaluating the CNN
# Includes data transforms, dataset loading, model initialization, and weight loading

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define preprocessing transformations for input images:
    # - Resize to 250x250 pixels to retain original image dimensions.
    #   This also guarantees all input images are the correct size, which prevents
    #   errors from any inconsistencies or outliers in image dimensions.
    # - Convert to tensor to make the image data compatible with PyTorch.
    # - Normalize RGB channels to have mean 0.5 and std 0.5 to center the data
    #   and ensure consistent input distribution for better training stability.
    transform = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Load pre-split training dataset using ImageFolder
    # Subfolders under 'Training' should be named 'fire' and 'nofire'
    train_dataset = ImageFolder(root="./data/ForestFireDataset/Training", transform=transform)

    # Load testing dataset from flat folder of images
    # Labels are inferred from filenames: files starting with 'fire' are labeled 0, others labeled 1
    test_dataset = TestDatasetFromFilenames("./data/ForestFireDataset/Testing", transform=transform)

    # Create DataLoader for the training dataset
    # Batch size is set to 32 to balance training speed, memory usage, and gradient stability
    # Shuffle is enabled to randomize input order and improve generalization
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Create DataLoader for the testing dataset
    # Batch size is 32 for consistency; shuffle is disabled to preserve original image order
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Instantiate and train the InitialCNN model
    initial_model = InitialCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(initial_model.parameters(), lr=0.001)
    train(initial_model, device, train_loader, criterion, optimizer)
    test(initial_model, device, test_loader)

    # Save model weights for transfer learning
    torch.save(initial_model.state_dict(), "initial_cnn_weights.pth")

    # Instantiate ExtendedCNN and load weights into its compatible conv_layer
    extended_model = ExtendedCNN().to(device)
    pretrained_weights = torch.load("initial_cnn_weights.pth")
    conv_weights = {k.replace("conv_layer.", ""): v for k, v in pretrained_weights.items() if k.startswith("conv_layer.")}
    extended_model.conv_layer.load_state_dict(conv_weights)

    # Freeze the pretrained convolutional layer weights so they are not updated during training
    for param in extended_model.conv_layer.parameters():
        param.requires_grad = False

    # Retrain or fine-tune ExtendedCNN if desired
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(extended_model.parameters(), lr=0.001)
    train(extended_model, device, train_loader, criterion, optimizer)
    test(extended_model, device, test_loader)

if __name__ == "__main__":
    main()
