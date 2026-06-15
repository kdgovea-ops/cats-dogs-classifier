import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
import os

# Set device (checking if GPU is available becuase it is faster)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data paths
TRAIN_DIR = 'data/train' # Where training images are
BATCH_SIZE = 32 # How any images trained at a time
EPOCHS = 10
LEARNING_RATE = 0.001

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), # Converting image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]) # Standardize pixel values to make training stable
])

# Load data
print("Loading data...")
train_dataset = ImageFolder(TRAIN_DIR, transform=transform) # ImageFolder() reads images
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # bundles batch size together

print(f"Dataset size: {len(train_dataset)}")
print(f"Classes: {train_dataset.classes}")

# Create model (ResNet18)
print("Creating model...")
model = models.resnet18(pretrained=True) # Loads pretrained ResNet18
model.fc = nn.Linear(512, 2)  # 2 classes: cats and dogs
model = model.to(device) # Moves to CPU

# Loss and optimizer
criterion = nn.CrossEntropyLoss() # Loss function
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Optimization algorithm, adjusts weights

# Training loop
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Save model
print("Saving model...")
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/cats_dogs_model.pth')
print("Model saved to models/cats_dogs_model.pth")