import os
import logging
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory

# Flask app configuration
app = Flask(__name__)

# Directory paths
dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = os.path.join(dir_path, "uploads")
STATIC_FOLDER = os.path.join(dir_path, "static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Image size for the model
IMAGE_SIZE = 192

# Load CNN model
try:
    cnn_model = tf.keras.models.load_model(os.path.join(STATIC_FOLDER, "models", "dog_cat_M.h5"))
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise e

# Preprocess an image
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0  # normalize to [0,1] range
    return image

# Read the image from path and preprocess
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

# Predict & classify image
def classify(model, image_path):
    try:
        preprocessed_image = load_and_preprocess_image(image_path)
        preprocessed_image = tf.reshape(preprocessed_image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

        prob = model.predict(preprocessed_image)
        label = "Cat" if prob[0][0] >= 0.5 else "Dog"
        classified_prob = prob[0][0] if prob[0][0] >= 0.5 else 1 - prob[0][0]

        return label, classified_prob
    except Exception as e:
        logger.error(f"Error during classification: {e}")
        return "Error", 0.0

# Home page
@app.route("/")
def home():
    return render_template("home.html")

# Image upload and classification
@app.route("/classify", methods=["POST", "GET"])
def upload_file():
    if request.method == "GET":
        return render_template("home.html")
    else:
        # Save uploaded file
        file = request.files.get("image")
        if not file:
            logger.error("No file uploaded.")
            return render_template("home.html", error="No file uploaded.")
        
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        try:
            file.save(upload_image_path)
            logger.info(f"File saved at {upload_image_path}")

            # Classify the image
            label, prob = classify(cnn_model, upload_image_path)
            prob = round(prob * 100, 2)

            return render_template(
                "classify.html", image_file_name=file.filename, label=label, prob=prob
            )
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return render_template("home.html", error="Error processing file.")

# Serve classified image
@app.route("/classify/<filename>")
def send_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        logger.error(f"Error sending file: {e}")
        return "Error sending file."

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
