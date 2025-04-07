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

# Custom Dataset to load test images and extract labels from filenames
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
        label = 0 if os.path.basename(image_path).lower().startswith("fire") else 1  # fire=0, nofire=1
        if self.transform:
            image = self.transform(image)
        return image, label

def train(model, device, train_loader, criterion, optimizer):
    # Set the model to training mode
    model.train()
    for epoch in range(10):
        # Initialize running loss and accuracy counters
        running_loss = 0.0
        correct, total = 0, 0
        
        # Iterate over each batch in the training data
        for inputs, labels in train_loader:
            # Move input data and labels to the selected device (CPU or GPU)
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients from the previous step
            optimizer.zero_grad()

            # Forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(inputs)

            # Compute the loss between predicted and actual labels
            loss = criterion(outputs, labels)

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Perform a single optimization step (parameter update)
            optimizer.step()

            # Accumulate the training loss
            running_loss += loss.item()

            # Get the index of the max log-probability (predicted class)
            _, predicted = torch.max(outputs, 1)

            # Update total number of labels and count of correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Compute average loss and accuracy for the epoch
        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total

        # Print epoch statistics
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

def test(model, device, test_loader):
    # Set the model to evaluation mode
    model.eval()
    # Initialize counters for correct predictions and total samples
    correct, total = 0, 0
    # Disable gradient calculation for inference
    with torch.no_grad():
        # Iterate over each batch in the test data
        for inputs, labels in test_loader:
            # Move input data and labels to the selected device (CPU or GPU)
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass: compute model outputs
            outputs = model(inputs)
            # Get the predicted class with highest score
            _, predicted = torch.max(outputs, 1)
            # Update total number of labels and count of correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # Print test accuracy based on predictions
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

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

    model = InitialCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, device, train_loader, criterion, optimizer)
    test(model, device, test_loader)

    # Save model weights for transfer learning
    torch.save(model.state_dict(), "initial_cnn_weights.pth")
    
if __name__ == "__main__":
    main()
