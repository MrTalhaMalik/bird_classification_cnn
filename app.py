from flask import Flask, request, render_template, redirect, url_for, flash
import os
import torch
from PIL import Image
from torchvision import transforms
from werkzeug.utils import secure_filename
from src.model import build_model  # Make sure this import matches your project structure

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'supersecretkey'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_class_names(data_directory):
    class_names = sorted(os.listdir(data_directory))
    return class_names

class_names = get_class_names('data/train')

def load_model(model_path, num_classes):
    model = build_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model('models/bird_classification_model.pth', len(class_names))

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def model_predict(image_path, model, class_names):
    preprocessed_image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(preprocessed_image)
    predicted_index = outputs.argmax().item()
    return class_names[predicted_index]

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            predicted_class_name = model_predict(file_path, model, class_names)
            return render_template('result.html', prediction=predicted_class_name, image_url=url_for('static', filename=os.path.join('uploads', filename)))
        else:
            flash('Allowed file types are png, jpg, jpeg')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True,port=5000, host='0.0.0.0')