# CNN_Image_Classifier_CIFAR-10
Designed and implemented a deep learning image classifier using a Convolutional Neural Network (CNN) with TensorFlow and Keras. The model was trained on the CIFAR-10 dataset, incorporating data preprocessing, feature extraction, and model evaluation techniques to classify images across 10 object categories with high accuracy.

<div align="center">

# 🧠 CIFAR-10 Image Classifier — ANN vs CNN

**A from-scratch comparison of a plain Artificial Neural Network and a Convolutional Neural Network on the CIFAR-10 dataset, built with TensorFlow/Keras.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-API-D00000?style=flat-square&logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

</div>

---

## 📌 Overview

This project trains and evaluates two image classification models on the **CIFAR-10** dataset — 60,000 32×32 color images across 10 classes — to directly demonstrate *why* Convolutional Neural Networks outperform plain fully-connected networks on image data.

1. A **baseline ANN** (dense layers only) for comparison
2. A **CNN** with convolution + pooling layers

The CNN raises test accuracy from **48% → 70%** over the ANN baseline using a small, simple architecture — no transfer learning, no heavy augmentation, just the core building blocks done right.

## 🖼️ Sample Data

<div align="center">
<img src="assets/sample_data.png" alt="Sample CIFAR-10 image" width="220"/>
</div>

CIFAR-10 contains 10 classes: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck` — 6,000 images per class, 50,000 for training and 10,000 for testing.

## 🏗️ Model Architectures

### Baseline: Artificial Neural Network (ANN)

```python
models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(3000, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='sigmoid'),
])
# optimizer='SGD', loss='sparse_categorical_crossentropy'
```

### Convolutional Neural Network (CNN)

```python
models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax'),
])
# optimizer='adam', loss='sparse_categorical_crossentropy'
```

## 📊 Results

| Model | Epochs | Train Accuracy | Test Accuracy |
|---|---|---|---|
| ANN (baseline) | 5 | 49.2% | 48% |
| **CNN** | 10 | 77.5% | **70%** |

> Swapping dense layers for convolution + pooling lifted test accuracy by **+22 points** with fewer training epochs — the core motivation for using CNNs on image data.

<details>
<summary>📋 Full classification report (CNN, test set)</summary>

```
              precision    recall  f1-score   support

   airplane       0.74      0.76      0.75      1000
 automobile       0.82      0.80      0.81      1000
       bird       0.53      0.67      0.59      1000
        cat       0.61      0.41      0.49      1000
       deer       0.64      0.64      0.64      1000
        dog       0.62      0.61      0.62      1000
       frog       0.78      0.77      0.77      1000
      horse       0.67      0.81      0.73      1000
       ship       0.84      0.76      0.80      1000
      truck       0.79      0.76      0.77      1000

    accuracy                           0.70     10000
   macro avg       0.70      0.70      0.70     10000
weighted avg       0.70      0.70      0.70     10000
```

</details>

### Predictions

<table>
<tr>
<td align="center"><b>✅ Correct</b><br/><img src="assets/correct_prediction.png" width="160"/><br/>Predicted: <code>ship</code></td>
<td align="center"><b>❌ Incorrect</b><br/><img src="assets/incorrect_prediction.png" width="160"/><br/>Predicted: <code>deer</code> (actual: <code>frog</code>)</td>
</tr>
</table>

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras** — model building and training
- **NumPy** — array operations
- **Matplotlib** — visualizing samples and predictions
- **scikit-learn** — classification report and confusion matrix

## 📂 Project Structure

```
cifar10-cnn/
├── Image_classifier_CNN.ipynb   # Main notebook: data loading, ANN, CNN, evaluation
├── assets/                      # Sample images used in this README
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### Run

1. Clone this repository
2. Open `Image_classifier_CNN.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab
3. Run all cells — CIFAR-10 downloads automatically via `tensorflow.keras.datasets.cifar10`

## 🔮 Future Improvements

- [ ] Add data augmentation (rotation, flips, zoom) to reduce overfitting
- [ ] Experiment with deeper architectures and Batch Normalization / Dropout
- [ ] Try transfer learning with a pretrained backbone (e.g., ResNet, MobileNet)
- [ ] Add a confusion matrix heatmap for per-class error analysis
- [ ] Package as a simple web app for live image upload + prediction

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Piyush**
[LinkedIn](https://www.linkedin.com/in/piyush-7658a3249)