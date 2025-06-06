import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def extract_features(img_path):
    try:
        # Baca gambar dengan mendukung path panjang/spasi
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Gambar tidak terbaca")
            
        # Preprocessing
        img = cv2.resize(img, (100, 100))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Ekstraksi fitur
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()
        edges = cv2.Canny(img_gray, 50, 150).flatten()
        
        return np.concatenate([hist, edges])
    except Exception as e:
        print(f"Error processing {os.path.basename(img_path)}: {str(e)}")
        return None

# Path handling yang kompatibel
dataset_path = os.path.abspath("dataset/train")
categories = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

X, y = [], []
for label, category in enumerate(categories):
    category_path = os.path.join(dataset_path, category)
    print(f"Processing: {category}")
    
    for img_file in os.listdir(category_path):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(category_path, img_file)
            features = extract_features(img_path)
            if features is not None:
                X.append(features)
                y.append(label)

# Konversi ke numpy array
X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Random Forest
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluasi
accuracy = model.score(X_test, y_test)
print(f"\nAccuracy: {accuracy:.2%}")

# Simpan model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/leaf_classifier.pkl")
joblib.dump(categories, "model/label_mapping.pkl")
print("Model saved to model/leaf_classifier.pkl")