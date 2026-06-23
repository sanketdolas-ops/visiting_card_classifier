"""
Version 3 trainer for the custom document classifier.

Default input:
    visiting_card_classifier/Training module_3/

Default output:
    visiting_card_classifier/trained_model_3/

Main upgrades over v2:
    - Multi-label classification support
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3
from tqdm.auto import tqdm
from tqdm.keras import TqdmCallback


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


BASE_DIR = Path(__file__).resolve().parent
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"}

CONFIG = {
    "model_version": "v3",
    "categories": [
        "Digital",
        "Handwritten",
        "Physical",
        "Rx Pad",
        "Stamp",
    ],
    "backbone": "EfficientNetB0",
    "image_size": [224, 224],
    "batch_size": 32,
    "test_split": 0.10,
    "validation_split": 0.20,
    "head_epochs": 8,
    "fine_tune_epochs": 10,
    "head_learning_rate": 1e-3,
    "fine_tune_learning_rate": 1e-5,
    "fine_tune_last_layers": 80,
    "dropout_1": 0.45,
    "dropout_2": 0.25,
    "l2_strength": 1e-4,
    "random_state": 42,
    "confidence_threshold": 0.50, # Adjusted for multi-label
}


def get_backbone_class(name):
    if name == "EfficientNetB3":
        return EfficientNetB3
    return EfficientNetB0


def collect_image_paths_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    # The 'labels' column is expected to have comma-separated labels
    df['labels'] = df['labels'].apply(lambda x: [label.strip() for label in x.split(',')])
    return df


def split_dataset(df):
    # For multi-label, we can't stratify directly on the list of labels.
    # A common approach is to use iterative stratification, but for simplicity,
    # we'll do a random split here.
    train_val, test = train_test_split(
        df,
        test_size=CONFIG["test_split"],
        random_state=CONFIG["random_state"],
    )

    relative_val_split = CONFIG["validation_split"] / (1.0 - CONFIG["test_split"])
    train, val = train_test_split(
        train_val,
        test_size=relative_val_split,
        random_state=CONFIG["random_state"],
    )

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def load_images(df, split_name="images"):
    height, width = CONFIG["image_size"]
    num_classes = len(CONFIG["categories"])
    cat_to_index = {cat: i for i, cat in enumerate(CONFIG["categories"])}
    images = []
    labels = []
    kept_rows = []

    rows = df.to_dict("records")
    for row in tqdm(rows, desc=f"Loading {split_name}", unit="image"):
        image_path = row["image_path"]
        img = cv2.imread(str(BASE_DIR / image_path))
        if img is None:
            print(f"WARNING: Could not read image: {image_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, height))
        images.append(img.astype(np.float32))
        
        # Create one-hot encoded label
        label_vector = np.zeros(num_classes, dtype=np.float32)
        for label in row['labels']:
            if label in cat_to_index:
                label_vector[cat_to_index[label]] = 1.0
        labels.append(label_vector)
        
        kept_rows.append(row)

    return np.array(images), np.array(labels), pd.DataFrame(kept_rows)


def make_augmentation():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def build_model():
    height, width = CONFIG["image_size"]
    num_classes = len(CONFIG["categories"])
    backbone_class = get_backbone_class(CONFIG["backbone"])

    inputs = keras.Input(shape=(height, width, 3))
    x = make_augmentation()(inputs)

    backbone = backbone_class(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )
    backbone.trainable = False

    x = layers.GlobalAveragePooling2D(name="avg_pool")(backbone.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(
        256,
        activation="    ",
        kernel_regularizer=regularizers.l2(CONFIG["l2_strength"]),
    )(x)
    x = layers.Dropout(CONFIG["dropout_1"])(x)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(CONFIG["l2_strength"]),
    )(x)
    x = layers.Dropout(CONFIG["dropout_2"])(x)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="category")(x)

    model = keras.Model(inputs, outputs, name=f"document_classifier_{CONFIG['model_version']}")
    return model, backbone


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name='auc')],
    )


def make_callbacks(output_dir, stage_name):
    return [
        TqdmCallback(verbose=1),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
        ),
        keras.callbacks.ModelCheckpoint(
            str(Path(output_dir) / f"best_{stage_name}.h5"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
    ]


def fine_tune_backbone(backbone):
    backbone.trainable = True
    cutoff = max(0, len(backbone.layers) - CONFIG["fine_tune_last_layers"])

    for layer in backbone.layers[:cutoff]:
        layer.trainable = False

    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def plot_history(head_history, fine_history, output_dir):
    history = {}
    for key in head_history.history:
        history[key] = head_history.history[key] + fine_history.history.get(key, [])

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    axes[0].plot(history.get("accuracy", []), label="Train")
    axes[0].plot(history.get("val_accuracy", []), label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.get("loss", []), label="Train")
    axes[1].plot(history.get("val_loss", []), label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(history.get("auc", []), label="Train")
    axes[2].plot(history.get("val_auc", []), label="Validation")
    axes[2].set_title("AUC")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / "training_history_v3.png"), dpi=120)
    plt.close()


def save_dataframe_to_locations(df, filename, output_dir, report_dir=None, excel=False, index=False):
    locations = [Path(output_dir)]
    if report_dir:
        report_path = Path(report_dir)
        if report_path.resolve() != Path(output_dir).resolve():
            locations.append(report_path)

    for location in locations:
        location.mkdir(parents=True, exist_ok=True)
        if excel:
            df.to_excel(location / filename, index=False)
        else:
            df.to_csv(location / filename, index=index)


def save_reports(model, X_test, y_test, test_rows, output_dir, report_dir=None):
    print("\nRunning final test predictions...")
    probabilities = model.predict(
        X_test,
        batch_size=CONFIG["batch_size"],
        verbose=0,
        callbacks=[TqdmCallback(verbose=1)],
    )
    predicted = (probabilities > CONFIG["confidence_threshold"]).astype(int)

    categories = CONFIG["categories"]
    
    # For multi-label, accuracy can be calculated in various ways.
    # Here, we calculate subset accuracy (exact match).
    test_accuracy = float(np.mean(np.all(predicted == y_test, axis=1)))
    print(f"\nTest Subset Accuracy (Exact Match): {test_accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predicted, target_names=categories, zero_division=0))

    report = classification_report(
        y_test,
        predicted,
        target_names=categories,
        zero_division=0,
        output_dict=True,
    )
    save_dataframe_to_locations(
        pd.DataFrame(report).transpose(),
        "classification_report_v3.csv",
        output_dir,
        report_dir,
        index=True,
    )

    analysis = test_rows.copy()
    y_test_labels = [[categories[i] for i, v in enumerate(row) if v] for row in y_test]
    predicted_labels = [[categories[i] for i, v in enumerate(row) if v] for row in predicted]

    analysis["actual_category"] = [",".join(labels) for labels in y_test_labels]
    analysis["predicted_category"] = [",".join(labels) for labels in predicted_labels]
    analysis["remarks"] = ""
    for index, category in enumerate(categories):
        analysis[f"score_{category}"] = np.round(probabilities[:, index] * 100, 2)

    save_dataframe_to_locations(
        analysis,
        "test_predictions_v3.csv",
        output_dir,
        report_dir,
    )
    errors = analysis[analysis["actual_category"] != analysis["predicted_category"]]
    save_dataframe_to_locations(
        errors,
        "error_analysis_v3.csv",
        output_dir,
        report_dir,
    )
    save_dataframe_to_locations(
        errors,
        "error_analysis_v3.xlsx",
        output_dir,
        report_dir,
        excel=True,
    )

    return test_accuracy


def save_artifacts(model, output_dir, training_folder, test_accuracy):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_file = output_path / "visiting_card_model_v3.h5"
    model.save(str(model_file))

    class_index = {category: index for index, category in enumerate(CONFIG["categories"])}
    with open(output_path / "class_index.json", "w", encoding="utf-8") as f:
        json.dump(class_index, f, indent=2)

    metadata = {
        **CONFIG,
        "training_folder": str(training_folder),
        "model_file": model_file.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "test_accuracy": round(test_accuracy * 100, 2),
    }
    with open(output_path / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model to: {model_file}")
    print(f"Saved metadata to: {output_path / 'model_metadata.json'}")


def train(training_folder, labels_csv, output_dir, report_dir=None):
    training_folder = Path(training_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"DOCUMENT CLASSIFIER TRAINING {CONFIG['model_version']}")
    print("=" * 72)
    print(f"Labels CSV: {labels_csv}")
    print(f"Output model folder: {output_dir}")
    print(f"Backbone: {CONFIG['backbone']}")
    print(
        "Training plan: "
        f"{CONFIG['head_epochs']} head epochs + "
        f"{CONFIG['fine_tune_epochs']} fine-tune epochs "
        f"(batch size {CONFIG['batch_size']})"
    )

    df = collect_image_paths_from_csv(labels_csv)
    train_df, val_df, test_df = split_dataset(df)

    train_df.to_csv(output_dir / "train_split_v3.csv", index=False)
    val_df.to_csv(output_dir / "validation_split_v3.csv", index=False)
    test_df.to_csv(output_dir / "test_split_v3.csv", index=False)

    X_train, y_train, train_rows = load_images(train_df, "training images")
    X_val, y_val, val_rows = load_images(val_df, "validation images")
    X_test, y_test, test_rows = load_images(test_df, "test images")

    print(f"Train images: {len(X_train)}")
    print(f"Validation images: {len(X_val)}")
    print(f"Test images: {len(X_test)}")
    
    # Note: class_weight is not used here as it's more complex for multi-label.
    # It can be added back if needed.

    model, backbone = build_model()

    print("\nStage 1: training classifier head")
    compile_model(model, CONFIG["head_learning_rate"])
    head_history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG["head_epochs"],
        batch_size=CONFIG["batch_size"],
        callbacks=make_callbacks(output_dir, "head"),
        verbose=0,
    )

    print("\nStage 2: fine-tuning last backbone layers")
    fine_tune_backbone(backbone)
    compile_model(model, CONFIG["fine_tune_learning_rate"])
    fine_history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG["fine_tune_epochs"],
        batch_size=CONFIG["batch_size"],
        callbacks=make_callbacks(output_dir, "fine_tune"),
        verbose=0,
    )

    plot_history(head_history, fine_history, output_dir)
    test_accuracy = save_reports(model, X_test, y_test, test_rows, output_dir, report_dir)
    save_artifacts(model, output_dir, str(training_folder), test_accuracy)

    print("\nTraining complete.")
    return test_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Train version 3 multi-label document classifier.")
    parser.add_argument(
        "--training-folder",
        default=str(BASE_DIR / "Training module_3"),
        help="Folder containing category subfolders. Used for resolving relative paths in CSV.",
    )
    parser.add_argument(
        "--labels-csv",
        required=True,
        help="Path to the CSV file with image paths and labels.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BASE_DIR / "trained_model_3"),
        help="Folder where the trained model and reports will be saved.",
    )
    parser.add_argument(
        "--backbone",
        choices=["EfficientNetB0", "EfficientNetB3"],
        default=CONFIG["backbone"],
        help="Use EfficientNetB3 only if your machine can handle slower training.",
    )
    parser.add_argument(
        "--report-dir",
        default=str(BASE_DIR / "sample_cards"),
        help="Folder where confusion matrix and error-analysis files are also saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CONFIG["backbone"] = args.backbone
    if args.backbone == "EfficientNetB3":
        CONFIG["image_size"] = [300, 300]
        CONFIG["batch_size"] = 16

    train(args.training_folder, args.labels_csv, args.output_dir, args.report_dir)
