import argparse
import json
from datetime import datetime
from pathlib import Path

import os
import shutil
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
    
    batch_size = CONFIG.get("batch_size", 32)
    results = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Predicting images"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        valid_paths = []

        for image_path in batch_paths:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"WARNING: Could not read image: {image_path}")
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, tuple(CONFIG['image_size']))
            batch_images.append(img.astype(np.float32))
            valid_paths.append(image_path)
        
        if not batch_images:
            continue

        all_probabilities = model.predict(np.array(batch_images), verbose=0)
        
        for image_path, probabilities in zip(valid_paths, all_probabilities):
            predicted_labels_indices = np.where(probabilities > CONFIG['confidence_threshold'])[0]
            predicted_labels = [CONFIG['categories'][i] for i in predicted_labels_indices]
            
            row = {'image_name': image_path.name}
            for cat_idx, category in enumerate(CONFIG['categories']):
                row[f'score_{category}'] = np.round(probabilities[cat_idx] * 100, 2)
            
            row['predicted_category'] = ",".join(predicted_labels) if predicted_labels else "Unclassified"
            row['confidence'] = np.max(probabilities) * 100 if predicted_labels else 0
            results.append(row)
        
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to: {output_csv}")

    return df

def organize_files(df: pd.DataFrame, source_folder: str):
    """
    Organizes classified files into subdirectories based on the V3 prediction results.
    """
    print("\nOrganizing files into subdirectories...")
    source_path = Path(source_folder)
    output_base = source_path / "organized_output_v3"
    
    moved_count = 0
    error_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Organizing files"):
        try:
            source_file = source_path / row["image_name"]
            if not source_file.exists():
                continue

            # Replace commas and spaces for a valid folder name
            predicted_folder = row["predicted_category"].replace(",", "_").replace(" ", "")
            dest_dir = output_base / predicted_folder
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_file), str(dest_dir / source_file.name))
            moved_count += 1
        except Exception as e:
            print(f"Error moving {row['image_name']}: {e}")
            error_count += 1
    
    print(f"\n✓ File organization complete. Output folder: {output_base.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction on a folder of images.")
    parser.add_argument("--model-path", required=True, help="Path to the trained model (.h5 file).")
    parser.add_argument("--image-folder", required=True, help="Path to the folder of images.")
    parser.add_argument("--output-csv", required=True, help="Path to save the output CSV file.")
    args = parser.parse_args()
    
    results_df = predict(args.model_path, args.image_folder, args.output_csv)
    
    if not results_df.empty:
        organize_files(results_df, args.image_folder)