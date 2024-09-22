from flask import Flask, request, render_template, send_file
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import base64

# Initialize the Flask app
app = Flask(__name__)

# Load your pre-trained U-Net model
MODEL_PATH = "unet_model_satalite images.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define a route to render the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error="No selected file")

    # Open and preprocess the image
    img = Image.open(file)
    original_image = img.copy()
    img = img.resize((128, 128))

    img_array = np.array(img) / 255.0

    if img_array.shape[-1] == 3:
        img_array = np.pad(img_array, ((0, 0), (0, 0), (0, 9)), mode='constant')

    img_array = np.expand_dims(img_array, axis=0)

    # Make the prediction
    prediction = model.predict(img_array)
    predicted_mask = (prediction[0] > 0.5).astype(np.uint8)

    # Convert the predicted mask back to an image
    predicted_mask_img = Image.fromarray(predicted_mask.squeeze() * 255)

    # Apply water color effect
    water_color_img = apply_water_color_effect(predicted_mask)

    # Convert images to base64 for display in HTML
    original_image_url = image_to_base64(original_image)
    predicted_mask_url = image_to_base64(predicted_mask_img)
    water_color_url = image_to_base64(water_color_img)

    # Return the rendered template with the predictions
    return render_template('index.html',
                           original_image_url=original_image_url,
                           predicted_mask_url=predicted_mask_url,
                           water_color_url=water_color_url)

def apply_water_color_effect(mask):
    # Create a color map (blue for water)
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_mask[mask.squeeze() == 1] = [0, 0, 255]  # Blue for water

    return Image.fromarray(color_mask)

def image_to_base64(image):
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    return f"data:image/png;base64,{base64.b64encode(img_io.getvalue()).decode('utf-8')}"

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
