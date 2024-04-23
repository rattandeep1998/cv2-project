import pandas as pd
from ChartDataset import ChartDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader

# Load the CSV file
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    # Map 'id' to the filename and 'type' to the label
    id_label_map = {row['id']: row['type'] for index, row in data.iterrows()}
    return id_label_map

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_labels = load_data('../../data/chartqa/train/unichart_formatted_chart_type_annotations.csv')
class_names = sorted(set(train_labels.values()))
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

# Assuming that training dataset contains all the classes
val_labels = load_data('../../data/chartqa/val/unichart_formatted_chart_type_annotations.csv')
test_labels = load_data('../../data/chartqa/test/unichart_formatted_chart_type_annotations.csv')
print(f"Class to Index Mapping: {class_to_idx}")

# Initialize dataset
train_dataset = ChartDataset('../../data/chartqa/train/png', train_labels, class_to_idx, transform)
val_dataset = ChartDataset('../../data/chartqa/val/png', val_labels, class_to_idx, transform)
test_dataset = ChartDataset('../../data/chartqa/test/png', test_labels, class_to_idx, transform)

# DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

num_classes = len(class_names)

# Print length of datasets
print(f'Number of training samples: {len(train_dataset)}')
print(f'Number of validation samples: {len(val_dataset)}')
print(f'Number of test samples: {len(test_dataset)}')

# Print class names
print(f"Total Classes: {num_classes}")
print(f"Training Dataset Classes: {train_dataset.classes}")
print(f"Validation Dataset Classes: {val_dataset.classes}")
print(f"Test Dataset Classes: {test_dataset.classes}")
      
# Load a pre-trained model
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)  # Assuming number of classes is known
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, best_val_loss=float('inf'), num_epochs=10, model_save_path='./best_model.pth'):
    best_val_loss = best_val_loss
    for epoch in range(num_epochs):
        running_train_loss = 0.0
        model.train()  # Set model to training mode
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            

        train_loss = running_train_loss / len(train_loader)

        val_loss, val_acc = validate_model(model, val_loader, criterion)
        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Save the model if the validation loss decreased
        if val_loss < best_val_loss:
            print(f'Validation loss decreased from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...')
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)

# Validation function
def validate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels)
            running_loss += loss.item()
    accuracy = correct_predictions.double() / len(dataloader.dataset)
    return running_loss / len(dataloader), accuracy

# TRAINING AND VALIDATION

# Save initial model before training
model_save_path='./best_resnet_model.pth'
val_loss, val_acc = validate_model(model, val_loader, criterion)
print(f'Epoch 0: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
torch.save(model.state_dict(), model_save_path)

train_and_validate(model, train_loader, val_loader, criterion, optimizer, best_val_loss=val_loss, model_save_path=model_save_path)

# TESTING
def load_best_model(model_path, num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model

best_model_path = model_save_path
best_model = load_best_model(best_model_path, num_classes)

def test_model(model, test_loader, criterion):
    model.eval()
    total_test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            print(labels)
            print("=============")
            correct += (predicted == labels).sum().item()

    average_test_loss = total_test_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    print(f'Test Loss: {average_test_loss:.4f}, Test Accuracy: {accuracy:.4f}%')

test_model(best_model, test_loader, criterion)