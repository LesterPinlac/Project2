import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define device (MPS for Mac, else CPU)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Define the CustomCNN Model
class CustomCNN(torch.nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Adjusted Convolutional Block 1
        self.conv_block1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )
        # Convolutional Block 2
        self.conv_block2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )
        # Convolutional Block 3
        self.conv_block3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )
        # Convolutional Block 4
        self.conv_block4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )
        # Adaptive Pooling
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # Fully Connected Layer with adjusted dropout
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Instantiate and load the trained model
model = CustomCNN().to(device)
model.load_state_dict(torch.load("best_custom_cnn.pth", map_location=device))
model.eval()

# Define preprocessing transformations (consistent with training)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to preprocess a single image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return test_transforms(image).unsqueeze(0).to(device)

# Function to predict the class of an image and return probabilities
def predict_image(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    return probabilities

# Paths to test images
test_images = [
    "test/crack/test_crack.jpg",
    "test/missing-head/test_missinghead.jpg",
    "test/paint-off/test_paintoff.jpg"
]

# Class names (update if necessary based on your dataset)
class_names = ["Crack", "Missing Head", "Paint Off"]

# Plotting all test images in a single row
plt.figure(figsize=(15, 5))  # Adjust figure size for readability
for idx, image_path in enumerate(test_images):
    # Get probabilities for the image
    probabilities = predict_image(image_path)
    predicted_class_index = np.argmax(probabilities)
    predicted_class_name = class_names[predicted_class_index]

    # Load and plot the image
    original_image = Image.open(image_path)
    ax = plt.subplot(1, len(test_images), idx + 1)
    ax.imshow(original_image)
    ax.axis('off')

    # Display probabilities as text
    text = "\n".join([f"{class_names[i]}: {probabilities[i]*100:.1f}%" for i in range(len(class_names))])
    ax.set_title(f"Predicted: {predicted_class_name}", fontsize=12, color="green")
    ax.text(0.5, -0.15, text, fontsize=10, color="green", ha="center", transform=ax.transAxes)

# Show the final plot
plt.tight_layout()
plt.show()