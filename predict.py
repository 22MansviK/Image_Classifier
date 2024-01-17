# Import modules
import torch
from torchvision import models
import argparse
from PIL import Image
import numpy as np
import json

# Define command line arguments
parser = argparse.ArgumentParser(description='Predict flower name from an image')
parser.add_argument('image_path', type=str, help='Path to the image')
parser.add_argument('checkpoint', type=str, help='Path to the checkpoint')
parser.add_argument('--top_k', type=int, default=3, help='Return top K most likely classes')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Use a mapping of categories to real names')
parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
args = parser.parse_args()

# Load the checkpoint
checkpoint = torch.load(args.checkpoint)
print(checkpoint.keys())

# Load the model
model = getattr(models, checkpoint['arch'])(pretrained=True)
model.classifier = checkpoint['classifier']
model.load_state_dict(checkpoint['state_dict'])
model.class_to_idx = checkpoint['class_to_idx']

# Move model to device
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)

# Process the image
def process_image(image):
    
    # Resize the image
    width, height = image.size
    ratio = width / height
    if width <= height:
        new_width = 256
        new_height = int(new_width / ratio)
    else:
        new_height = 256
        new_width = int(new_height * ratio)
    image = image.resize((new_width, new_height))
    
    # Crop the image
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))
    
    # Convert to numpy array and normalize
    np_image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Transpose the array
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

# Predict the class
def predict(image_path, model, topk):
    
    # Open the image
    image = Image.open(image_path)
    
    # Process the image
    image = process_image(image)
    
    # Convert to tensor
    image = torch.from_numpy(image).type(torch.FloatTensor)
    
    # Add batch dimension
    image = image.unsqueeze(0)
    
    # Move input to device
    image = image.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Turn off gradients
    with torch.no_grad():
        # Forward pass
        logps = model.forward(image)
        
        # Calculate probabilities
        ps = torch.exp(logps)
        
        # Get top probabilities and classes
        top_p, top_class = ps.topk(topk, dim=1)
        
        # Convert tensors to lists
        top_p = top_p.tolist()[0]
        top_class = top_class.tolist()[0]
        
        # Convert indices to classes
        idx_to_class = {value: key for key, value in model.class_to_idx.items()}
        top_class = [idx_to_class[x] for x in top_class]
        
        return top_p, top_class

# Get probabilities and classes
probs, classes = predict(args.image_path, model, args.top_k)

# Load category names
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

# Convert classes to names
names = [cat_to_name[x] for x in classes]

# Print results
print(f"Predicting flower name from image {args.image_path}...")
print(f"Top {args.top_k} most likely classes and their probabilities:")
for i in range(args.top_k):
    print(f"{i+1}. {names[i]}: {probs[i]*100:.2f}%")

