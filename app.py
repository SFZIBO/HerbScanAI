from flask import Flask, request, render_template, redirect, url_for
import os
import cv2
import numpy as np
import joblib

# Load model dan mapping label
model = joblib.load("model/leaf_classifier.pkl")
categories = joblib.load("model/label_mapping.pkl")

# Fungsi ekstraksi fitur (harus sama dengan saat training)
def extract_features(img_path):
    try:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Gambar tidak terbaca")
        
        img = cv2.resize(img, (100, 100))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()
        edges = cv2.Canny(img_gray, 50, 150).flatten()
        
        return np.concatenate([hist, edges])
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Inisiasi Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            features = extract_features(filepath)
            if features is not None:
                prediction = model.predict([features])[0]
                predicted_label = categories[prediction]

                return redirect(url_for('result', image=file.filename, label=predicted_label))
    
    return render_template('index.html')

@app.route('/result')
def result():
    image = request.args.get('image')
    label = request.args.get('label')
    return render_template('result.html', image=image, label=label)

if __name__ == '__main__':
    app.run(debug=True)