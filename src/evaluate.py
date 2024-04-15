import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import build_model  # Ensure this imports your model architecture correctly

def evaluate(model_path: str, test_dir: str, img_height: int = 224, img_width: int = 224, batch_size: int = 32) -> None:
    """
    Evaluates the trained model on the test dataset using PyTorch.

    Args:
        model_path (str): Path to the saved model's state dictionary.
        test_dir (str): Path to the test directory containing images.
        img_height (int): Height of the input images. Default is 224.
        img_width (int): Width of the input images. Default is 224.
        batch_size (int): Size of the batches of data. Default is 32.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model = build_model(num_classes=525)  # Adjust num_classes as per your project's requirement
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Define transformation
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Adjust as per your normalization
    ])

    # Load the test data
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')