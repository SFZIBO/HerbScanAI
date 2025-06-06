import cv2
import joblib

# Load model dan mapping label
model = joblib.load("model_daun.pkl")
categories = joblib.load("label_mapping.pkl")

def predict_image(img_path):
    features = extract_features(img_path).reshape(1, -1)
    pred = model.predict(features)[0]
    return categories[pred]

# Contoh penggunaan
hasil = predict_image("test_daun.jpg")
print(f"Prediksi: {hasil}")