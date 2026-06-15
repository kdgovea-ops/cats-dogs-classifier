# src/validate.py
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load('models/cats_dogs_model.pth')) # Loads saved weights
model = model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Use training data for validation (split it)
train_dataset = ImageFolder('data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32)

# Evaluate
correct = 0
total = 0

with torch.no_grad(): # toorch.no_grad tells PyTorch to not track gradients, since we are predictig not training
    for images, labels in train_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Validation Accuracy on Training Data: {accuracy:.2f}%")