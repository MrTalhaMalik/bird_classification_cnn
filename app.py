from flask import Flask, request, render_template, redirect, url_for
import os
import torch
from PIL import Image
from torchvision import transforms
from src.model import build_model  # Update this import to your model's architecture file

app = Flask(__name__)

# Directory where uploaded images will be saved
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to get class names from the dataset directory
def get_class_names(data_directory):
    # List the directories and sort them alphabetically to maintain order
    class_names = sorted(os.listdir(data_directory))
    return class_names

# Assuming that the class names ordering is consistent with the training
class_names = get_class_names('data/train')

# Load your trained model
def load_model(model_path, num_classes):
    # Make sure the architecture is identical to the one used during training
    model = build_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Load the model (update the path as needed)
num_classes = len(class_names)  # Ensure this is the correct number of classes
model = load_model('models/bird_classification_model.pth', num_classes)

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    # Make sure these transformations match those used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to make a prediction with the model
def model_predict(image_path, model, class_names):
    preprocessed_image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(preprocessed_image)
    predicted_index = outputs.argmax().item()
    predicted_class_name = class_names[predicted_index]  # Map index to class name
    return predicted_class_name

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            # Make prediction
            predicted_class_name = model_predict(filename, model, class_names)
            # Render a template with the prediction result
            return render_template('prediction.html', prediction=predicted_class_name)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')