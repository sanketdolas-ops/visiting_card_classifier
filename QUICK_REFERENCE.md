# Model Training - Quick Reference Card

## 📊 ONE PAGE SUMMARY

### What We're Building
A **Deep Learning Model** that classifies images into 4 categories:
- Digital Cards (smooth, uniform)
- Handwritten Docs (irregular, pen marks)
- Rx Pads (medical format)
- Visiting Cards (professional layout)

---

## 🧠 The Model Architecture

```
IMAGE (224×224 pixels)
   ↓
EfficientNetB0 Backbone (pre-trained on 1.3M images)
   ↓
Custom Classifier Head (Dense layers)
   ↓
OUTPUT (4 probabilities)
```

**Why EfficientNetB0?**
- Fast training ⚡
- Accurate 🎯
- Lightweight 📦
- Perfect balance

---

## 📚 Training Process (Simple Steps)

### Step 1: Prepare Data
```
Load images from 4 folders:
  Digital/ + handwritten/ + Rx Pad/ + Visiting Card/
Resize to 224×224 pixels
Normalize pixel values (0-1)
```

### Step 2: Augment Data
```
Create variations:
  Rotate, Shift, Zoom, Flip
Result: 1 image → 4 variations
        50 images → 200 training samples
```

### Step 3: Split Data
```
70% Training (learns patterns)
15% Validation (checks progress)
15% Testing (final evaluation)
```

### Step 4: Train Model
```
For each epoch (up to 50):
  For each batch (32 images):
    1. Make predictions
    2. Calculate error (loss)
    3. Update weights
    4. Improve model
    
Stop early if validation stops improving
```

### Step 5: Save Model
```
Save:
  - weights (visiting_card_model.h5)
  - class mappings (class_index.json)
  - configuration (config.json)
  - history plot (training_history.png)
```

---

## 🔢 Key Metrics During Training

| Metric | Good | Bad |
|--------|------|-----|
| **Training Loss** | Decreasing | Stuck/increasing |
| **Val Loss** | Decreasing then stable | Increasing (overfitting) |
| **Training Acc** | Increasing to 90%+ | Stuck low |
| **Val Accuracy** | Increasing to 85%+ | Stuck low or decreasing |
| **Gap (Train-Val)** | Small (< 5%) | Large (> 10%) = overfitting |

---

## ⏱️ Typical Training Timeline

### Without GPU (CPU only)
```
Preparation:    0-1 minute
Loading data:   1-2 minutes
Training 50 epochs: 150-240 minutes (2.5-4 hours)
Total:          ~3-4 hours
```

### With GPU (NVIDIA CUDA)
```
Preparation:    0-1 minute
Loading data:    1 minute
Training 50 epochs: 20-40 minutes
Total:          ~25-45 minutes
```

### With Apple Silicon (M1/M2/M3)
```
Preparation:    0-1 minute
Loading data:    1 minute
Training 50 epochs: 15-30 minutes
Total:          ~20-35 minutes
```

---

## 📈 Expected Performance

After training on 50-200 images per category:

```
Digital:       92-98% accuracy
Rx Pad:        88-95% accuracy
Visiting Card: 85-92% accuracy
Handwritten:   80-90% accuracy

Overall:       88-94% accuracy
```

Depends on:
- Data quality ✓
- Category distinctiveness ✓
- Number of training images ✓

---

## 🎯 How It Differentiates

### Digital Cards
- Smooth texture ✓
- Uniform lighting ✓
- Sharp edges ✓
- Perfect colors ✓

### Handwritten
- Irregular strokes ✓
- Pen marks ✓
- Paper grain visible ✓
- Variable pressure ✓

### Rx Pads
- Medical format ✓
- Pre-printed elements ✓
- Specific layout ✓
- Professional header ✓

### Visiting Cards
- Card aspect ratio ✓
- Printed content ✓
- Logo/branding ✓
- Contact info ✓

---

## 🚀 Three Ways to Train

### Method 1: Quick Start (Recommended)
```bash
python quickstart.py
Select option 1
```

### Method 2: Direct Script
```bash
python train_custom_model.py
```

### Method 3: Python Code
```python
from train_custom_model import train_visiting_card_classifier

model, path = train_visiting_card_classifier(
    r"C:\...\Training module",
    r"C:\...\output"
)
```

---

## 📊 Monitoring Training

### Watch These Numbers

**Loss Graph** (should go DOWN)
```
Epoch 1:  Loss = 2.5 ❌
Epoch 10: Loss = 1.2 ⚠️
Epoch 20: Loss = 0.3 ✓
```

**Accuracy Graph** (should go UP)
```
Epoch 1:  Accuracy = 45% ❌
Epoch 10: Accuracy = 75% ⚠️
Epoch 20: Accuracy = 90% ✓
```

**Both curves** should be CLOSE
```
If train = 95% and val = 92% ✓ GOOD
If train = 98% and val = 60% ❌ OVERFITTING
```

---

## 🛑 Early Stopping Rule

```
If validation loss doesn't improve for 10 epochs:
   → STOP training
   → Restore best model
   
This prevents overfitting!
```

Example:
```
Epoch 15: Val Loss = 0.9 ← BEST
Epoch 16: Val Loss = 1.0
Epoch 17: Val Loss = 1.1
Epoch 18: Val Loss = 1.2
...
Epoch 25: Val Loss = 1.5 ← STOP! (10 epochs no improvement)
          Restore Epoch 15 model
```

---

## 💾 After Training: Files Generated

```
trained_model/
├── visiting_card_model.h5 (50-100 MB)
│   └─ Complete trained model
├── class_index.json
│   └─ {"Digital": 0, "handwritten": 1, ...}
├── config.json
│   └─ Training configuration
└── training_history.png
    └─ Accuracy and loss graphs
```

---

## 🔍 Using Trained Model

### Single Prediction
```python
from inference_custom_model import VisitingCardClassifier

classifier = VisitingCardClassifier(r"C:\...\trained_model")
result = classifier.predict(r"C:\...\test_image.jpg")

print(result)
# {'category': 'Digital', 'confidence': 92.3, ...}
```

### Batch Prediction
```python
df = classifier.predict_folder(
    r"C:\...\test_images",
    r"C:\...\predictions.xlsx"
)
print(df.groupby('category').size())
```

---

## ⚙️ Hyperparameters

| Parameter | Current | Meaning |
|-----------|---------|---------|
| batch_size | 32 | Images per update |
| epochs | 50 | Training rounds |
| learning_rate | 0.0001 | Step size for weight updates |
| image_size | 224×224 | Input pixels |
| dropout | 0.5, 0.3 | Overfitting prevention |
| augmentation | True | Create image variations |

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| No images found | Check folder structure |
| Out of memory | Reduce batch_size to 8-16 |
| Very slow | Use GPU or reduce epochs |
| Low accuracy | Add more training data |
| Model keeps improving past 50 epochs | Increase epochs |
| Accuracy stops improving | Already achieved optimal training |

---

## ✅ Checklist Before Training

- [ ] Python installed (3.8+)
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Training images organized in 4 folders
- [ ] Each category has 50+ images (recommended)
- [ ] Images are clear and representative
- [ ] Enough disk space (~500 MB for training)
- [ ] Ready to wait 30 min to 4 hours (depending on GPU)

---

## 🎓 Key Concepts Explained

### Transfer Learning
Using a model pre-trained on 1.3M images, just adapt it to your task.

### Backpropagation
Algorithm that figures out which weights to adjust to reduce error.

### Batch Processing
Process 32 images at a time, update weights after each batch.

### Epoch
One complete pass through all training data.

### Overfitting
Memorizing training data too well, fails on new data.

### Early Stopping
Stop training before overfitting happens.

### Dropout
Randomly disable neurons to prevent overfitting.

### Data Augmentation
Create variations (rotate, zoom) to make model robust.

---

## 📞 Quick Help

**Q: How long will training take?**
A: 30 min (GPU) to 4 hours (CPU)

**Q: How many images do I need?**
A: 50-200 per category recommended

**Q: Can I use my GPU?**
A: Yes, it will auto-detect CUDA/Metal

**Q: Will I lose my model if I stop training?**
A: No, early stopping saves the best model

**Q: Can I retrain on more data?**
A: Yes, just add more images to Training module/ and train again

**Q: What's the final accuracy?**
A: Typically 85-94% depending on data quality

---

## 🚀 Next Steps

1. Organize training images in 4 folders
2. Run `pip install -r requirements.txt`
3. Run `python quickstart.py`
4. Select option 1 to train
5. Wait for completion
6. Check `training_history.png`
7. Use trained model for predictions

**Ready to train!** 🎉
