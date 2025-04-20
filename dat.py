import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from torch.autograd import Function
from torchviz import make_dot
import matplotlib.pyplot as plt

# Gradient Reversal Layer definition
# This class defines a custom autograd function that performs identity in the forward pass
# and reverses the gradient during backpropagation to enable adversarial learning.
# This class is based on the Function class (torch.autograd.Function), which is used for
# defining operations that need custom operations for the forward and backward pass.
# The methods are defined using the @staticmethod decorator because they are overriding
# functions form the parent class that also have this decorator.
class GradientReversalLayer(Function):
  
    # Forward pass: returns the input as-is and stores lambda for use in the backward pass
    @staticmethod
    def forward(ctx, input, lambda_):
        ctx.lambda_ = lambda_
        return input.view_as(input)

    # Backward pass: reverses the gradient and scales it by lambda
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

# Wrapper module for the Gradient Reversal Layer
# This module integrates the gradient reversal mechanism as a standard PyTorch layer
class GRLWrapper(nn.Module):
    # Initializes the GRL wrapper with the given lambda value
    def __init__(self, lambda_):
        super(GRLWrapper, self).__init__()
        self.lambda_ = lambda_

    # Applies the gradient reversal function to the input
    def forward(self, x):
        return GradientReversalLayer.apply(x, self.lambda_)

# Domain Discriminator for adversarial domain classification
# This module is a binary classifier that predicts whether features come from the source or target domain
class DomainDiscriminator(nn.Module):
    # Initializes the discriminator network with the specified input feature dimension
    def __init__(self, input_dim=128*31*31):
        super(DomainDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    # Performs a forward pass through the domain discriminator
    def forward(self, x):
        return self.net(x)

# Domain Adversarial Trainer encapsulates the logic for adversarial domain adaptation
# It jointly trains a label classifier on source data and a domain discriminator on both source and target data
class DomainAdversarialTrainer:
    def __init__(self, feature_extractor, label_classifier, domain_discriminator, grl_layer, source_loader, target_loader, device):
        self.feature_extractor = feature_extractor
        self.label_classifier = label_classifier
        self.domain_discriminator = domain_discriminator
        self.grl = grl_layer
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.device = device
        self.label_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.BCELoss()
        self.optimizer = optim.Adam(list(self.feature_extractor.parameters()) +
                                    list(self.label_classifier.parameters()) +
                                    list(self.domain_discriminator.parameters()), lr=0.001)

    def train_epoch(self):
        self.feature_extractor.train()
        self.label_classifier.train()
        self.domain_discriminator.train()

        total_class_loss = 0.0
        total_domain_loss = 0.0
        correct, total = 0, 0

        target_iter = iter(self.target_loader)

        for source_data, source_labels in self.source_loader:
            try:
                target_data, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(self.target_loader)
                target_data, _ = next(target_iter)

            source_data, source_labels = source_data.to(self.device), source_labels.to(self.device)
            target_data = target_data.to(self.device)

            self.optimizer.zero_grad()

            # Extract features
            source_features = self.feature_extractor(source_data)
            target_features = self.feature_extractor(target_data)

            # Label prediction on source
            class_outputs = self.label_classifier(source_features)
            class_loss = self.label_criterion(class_outputs, source_labels)

            # Domain prediction on source and target
            domain_inputs = torch.cat([source_features, target_features], dim=0)
            domain_labels = torch.cat([torch.ones(source_features.size(0)), torch.zeros(target_features.size(0))], dim=0).to(self.device)
            domain_outputs = self.domain_discriminator(self.grl(domain_inputs)).view(-1)
            domain_loss = self.domain_criterion(domain_outputs, domain_labels)

            # Combine losses and backpropagate
            total_loss = class_loss + domain_loss
            total_loss.backward()
            self.optimizer.step()

            total_class_loss += class_loss.item()
            total_domain_loss += domain_loss.item()
            _, predicted = torch.max(class_outputs, 1)
            total += source_labels.size(0)
            correct += (predicted == source_labels).sum().item()

        avg_class_loss = total_class_loss / len(self.source_loader)
        avg_domain_loss = total_domain_loss / len(self.source_loader)
        accuracy = 100.0 * correct / total

        print(f"Class Loss: {avg_class_loss:.4f}, Domain Loss: {avg_domain_loss:.4f}, Accuracy: {accuracy:.2f}%")


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

# Global lists to store metrics for both models
initial_epoch_times = []
initial_epoch_losses = []
initial_epoch_accuracies = []
extended_epoch_times = []
extended_epoch_losses = []
extended_epoch_accuracies = []

# Custom dataset for unlabeled domain adaptation (target domain)
# This class loads all images under a root directory and ignores labels
class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        for subfolder in ["Fire", "No_Fire"]:
            folder_path = os.path.join(root_dir, subfolder)
            if os.path.isdir(folder_path):
                self.image_paths.extend([os.path.join(folder_path, f)
                                         for f in os.listdir(folder_path)
                                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # dummy label

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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Load testing dataset from flat folder of images
    # Labels are inferred from filenames: files starting with 'fire' are labeled 0, others labeled 1
    test_dataset = TestDatasetFromFilenames("./data/ForestFireDataset/Testing", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load labeled source domain using ImageFolder (FLAME Training)
    source_dataset = ImageFolder(root="./data/FLAME/Training", transform=transform)
    source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True)

    # Load unlabeled target domain using custom dataset (FLAME Test)
    target_dataset = UnlabeledImageDataset(root_dir="./data/FLAME/Test", transform=transform)
    target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)
        
    # Instantiate and train the InitialCNN model
    initial_model = InitialCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(initial_model.parameters(), lr=0.001)
    train(initial_model, device, train_loader, criterion, optimizer)
    test(initial_model, device, test_loader)

    # Store InitialCNN metrics
    global epoch_times, epoch_losses, epoch_accuracies
    initial_epoch_times.extend(epoch_times)
    initial_epoch_losses.extend(epoch_losses)
    initial_epoch_accuracies.extend(epoch_accuracies)

    # Save model weights for transfer learning
    torch.save(initial_model.state_dict(), "initial_cnn_weights.pth")

    # Save initial model for visualization in Netron
    # torch.save(initial_model, "initial_cnn_model.pt")

    # Instantiate ExtendedCNN and load weights into its compatible conv_layer
    extended_model = ExtendedCNN().to(device)
    pretrained_weights = torch.load("initial_cnn_weights.pth")
    conv_weights = {k.replace("conv_layer.", ""): v for k, v in pretrained_weights.items() if k.startswith("conv_layer.")}
    extended_model.conv_layer.load_state_dict(conv_weights)

    # Freeze the pretrained convolutional layer weights so they are not updated during training
    for param in extended_model.conv_layer.parameters():
        param.requires_grad = False

    # Save extended model for visualization in Netron
    # torch.save(extended_model, "extended_cnn_model.pt")

    # Generate Torchviz graph for model architecture
    #dummy_input = torch.randn(1, 3, 250, 250).to(device)
    #initial_output = initial_model(dummy_input)
    #make_dot(initial_output, params=dict(initial_model.named_parameters())).render("initial_cnn_architecture", format="png")

    #extended_output = extended_model(dummy_input)
    #make_dot(extended_output, params=dict(extended_model.named_parameters())).render("extended_cnn_architecture", format="png")

    # Perform domain adversarial training instead of standard training on ExtendedCNN
    grl_layer = GRLWrapper(lambda_=1.0)
    domain_discriminator = DomainDiscriminator().to(device)
    datrainer = DomainAdversarialTrainer(
        feature_extractor=nn.Sequential(extended_model.conv_layer, extended_model.additional_conv),
        label_classifier=extended_model.fc_layer,
        domain_discriminator=domain_discriminator,
        grl_layer=grl_layer,
        source_loader=source_loader,
        target_loader=target_loader,
        device=device
    )

    for epoch in range(10):
        print(f"Epoch {epoch+1} (Adversarial)")
        datrainer.train_epoch()

    # Store ExtendedCNN metrics
    extended_epoch_times.extend(epoch_times)
    extended_epoch_losses.extend(epoch_losses)
    extended_epoch_accuracies.extend(epoch_accuracies)

if __name__ == "__main__":
    main()

    # Plotting metrics for comparison
    epochs = list(range(1, 11))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, initial_epoch_times, label='InitialCNN', color='blue')
    plt.plot(epochs, extended_epoch_times, label='ExtendedCNN', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.title('Epoch Time Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, initial_epoch_losses, label='InitialCNN', color='blue')
    plt.plot(epochs, extended_epoch_losses, label='ExtendedCNN', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, initial_epoch_accuracies, label='InitialCNN', color='blue')
    plt.plot(epochs, extended_epoch_accuracies, label='ExtendedCNN', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Epoch Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
