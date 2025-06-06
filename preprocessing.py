import cv2
import numpy as np

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))  # Resize untuk efisiensi
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Ekstrak fitur warna dan tekstur
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    return hist