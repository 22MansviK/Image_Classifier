# train.py

# Import modules
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse
import os

# Define command line arguments
parser = argparse.ArgumentParser(description='Train a neural network on a dataset')
parser.add_argument('data_dir', type=str, help='Path to the data directory')
parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
parser.add_argument('--arch', type=str, default='vgg13', help='Model architecture')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
args = parser.parse_args()

# Define data transforms
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

# Load data
train_data = datasets.ImageFolder(args.data_dir + '/train', transform=train_transforms)
valid_data = datasets.ImageFolder(args.data_dir + '/valid', transform=valid_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

# Load model
model = getattr(models, args.arch)(pretrained=True)
print(model)

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

# Define classifier
classifier = nn.Sequential(nn.Linear(model.classifier[0].in_features, args.hidden_units),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(args.hidden_units, 102),
                           nn.LogSoftmax(dim=1))

model.classifier = classifier

# Define criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Move model to device
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)

# Train the model
print("Training the model...")
steps = 0
running_loss = 0
print_every = 10
for epoch in range(args.epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Clear the gradients
        optimizer.zero_grad()
        
        # Forward and backward passes
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Validation loop
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{args.epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()

# Save the checkpoint
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

checkpoint = {'arch': args.arch,
              'hidden_units': args.hidden_units,
              'state_dict': model.state_dict(),
              'class_to_idx': train_data.class_to_idx,
              'classifier': model.classifier}

torch.save(checkpoint, args.save_dir + '/checkpoint.pth')

print("Model trained and checkpoint saved.")


