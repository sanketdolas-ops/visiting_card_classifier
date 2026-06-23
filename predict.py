import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm

def predict(model_path, image_folder, output_csv):
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    model_dir = Path(model_path).parent
    metadata_path = model_dir / 'model_metadata.json'
    print(f"Loading metadata from: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    CONFIG = metadata
    
    image_paths = sorted(
        path for path in Path(image_folder).rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"}
    )
    
    print(f"Found {len(image_paths)} images to predict.")
    
    results = []
    
    for image_path in tqdm(image_paths, desc="Predicting images"):
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"WARNING: Could not read image: {image_path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, tuple(CONFIG['image_size']))
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        
        probabilities = model.predict(img, verbose=0)[0]
        
        predicted_labels_indices = np.where(probabilities > CONFIG['confidence_threshold'])[0]
        predicted_labels = [CONFIG['categories'][i] for i in predicted_labels_indices]
        
        row = {'image_name': image_path.name}
        for i, category in enumerate(CONFIG['categories']):
            row[f'score_{category}'] = np.round(probabilities[i] * 100, 2)
        
        row['predicted_category'] = ",".join(predicted_labels) if predicted_labels else "Unclassified"
        row['confidence'] = np.max(probabilities) * 100 if predicted_labels else 0
        
        results.append(row)
        
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction on a folder of images.")
    parser.add_argument("--model-path", required=True, help="Path to the trained model (.h5 file).")
    parser.add_argument("--image-folder", required=True, help="Path to the folder of images.")
    parser.add_argument("--output-csv", required=True, help="Path to save the output CSV file.")
    args = parser.parse_args()
    
    predict(args.model_path, args.image_folder, args.output_csv)
