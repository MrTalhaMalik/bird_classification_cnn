import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
from PIL import Image
from torchvision import transforms
from src.model import build_model

app = Flask(__name__)

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model_path = 'models/bird_classification_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(num_classes=525)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define the allowed file types function
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the prediction function
def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        prediction = predict(file_path)
        return render_template('result.html', filename=filename, prediction=prediction)
    else:
        return redirect(request.url)

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(debug=True, port=5000, host='0.0.0.0')