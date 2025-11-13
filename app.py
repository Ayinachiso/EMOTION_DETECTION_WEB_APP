# ------------------------------------------------------------
# app.py - Emotion Detection Web App (Flask + CNN Model)
# ------------------------------------------------------------
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
import sqlite3
import os
import uuid

# Suppress TensorFlow startup logs (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ------------------------------------------------------------
# 1. Flask setup
# ------------------------------------------------------------
app = Flask(__name__)

# Path to trained model
MODEL_PATH = "face_emotionModel.h5"

# Load model once (efficient)
model = load_model(MODEL_PATH)

# Emotion class labels (must match training order)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Configure upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ------------------------------------------------------------
# 2. Database setup (SQLite)
# ------------------------------------------------------------
def init_db():
    """Initialize SQLite database for predictions."""
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    emotion TEXT,
                    confidence REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )''')
    conn.commit()
    conn.close()


init_db()


# ------------------------------------------------------------
# 3. Routes
# ------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    """Main page – handles image upload and prediction."""
    if request.method == "POST":
        file = request.files.get("image")

        if not file:
            return render_template("index.html", prediction="No file uploaded.")

        # Secure & unique filename
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)

        try:
            file.save(filepath)
        except Exception as e:
            return render_template("index.html", prediction=f"Error saving file: {str(e)}")

        # Preprocess image safely
        try:
            img = image.load_img(filepath, target_size=(48, 48), color_mode="grayscale")
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
        except Exception:
            return render_template("index.html", prediction="Invalid image format.")

        # Predict emotion
        try:
            predictions = model.predict(img_array)
            emotion_index = np.argmax(predictions)
            predicted_emotion = emotion_labels[emotion_index]
            confidence = float(np.max(predictions))
        except Exception as e:
            return render_template("index.html", prediction=f"Model error: {str(e)}")

        # Save result to database
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("INSERT INTO predictions (filename, emotion, confidence) VALUES (?, ?, ?)",
                  (unique_name, predicted_emotion, confidence))
        conn.commit()
        conn.close()

        # Convert path for HTML template
        image_path = "/" + filepath.replace("\\", "/")

        return render_template("index.html",
                               prediction=f"{predicted_emotion} ({confidence*100:.1f}%)",
                               image_path=image_path)

    # GET request → load empty form
    return render_template("index.html", prediction=None)


@app.route("/history")
def history():
    """Displays a list of all past predictions."""
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT filename, emotion, confidence, timestamp FROM predictions ORDER BY id DESC")
    records = c.fetchall()
    conn.close()
    return render_template("history.html", records=records)


# ------------------------------------------------------------
# 4. Run the app
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)