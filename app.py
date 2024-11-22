from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import os

app = Flask(__name__)

# Load the pre-trained Keras model
MODEL_PATH = r"C:\Users\mural\Downloads\idp\lung_cancer_detection_model.h5"
model = load_model(MODEL_PATH)

# Define classes if needed
CLASSES = ['Lung squamous_cell_carcinoma', 'Lung_adenocarcinoma', 'Lung_benign_tissue']

# Function to preprocess image
def preprocess_image(img):
    # Preprocess the image as required by your model
    # For example, resize, normalize, etc.
    return img

# Function to make predictions
def predict(image_path):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img = image.img_to_array(img)
    img = preprocess_image(img)
    img = img.reshape(1, img_width, img_height, 3)
    prediction = model.predict(img)
    predicted_class = CLASSES[prediction.argmax()]
    return predicted_class

# Route to upload an image
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join('uploads', f.filename)
        f.save(file_path)
        prediction = predict(file_path)
        return render_template('result.html', prediction=prediction, image_path=file_path)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
