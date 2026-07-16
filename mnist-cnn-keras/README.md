# MNIST CNN (Keras/TensorFlow)

A tiny convolutional neural network that hits **~98.28% test accuracy** on the classic MNIST handwritten digits dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KirilShy/AI-ML/blob/main/mnist_project.ipynb)

---

## Dataset
We use the **MNIST Handwritten Digits** dataset (60,000 training + 10,000 test images).  
📂 Download from: https://www.kaggle.com/datasets/hojjatk/mnist-dataset
Or load directly via TensorFlow/Keras:  
```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

---

## What’s inside

**Architecture**
- Input: 28×28 grayscale, normalized to [0,1]
- Conv2D(32, 3×3, ReLU) → MaxPool(2×2)  
- Conv2D(64, 3×3, ReLU) → MaxPool(2×2)  
- Flatten → Dense(64, ReLU) → Dense(10, Softmax)

**Training**
- Optimizer: **Adam**
- Loss: **sparse_categorical_crossentropy** (integer labels 0–9)
- Batch size: **64**, Epochs: **5**
- Validation split: **0.1**

**Results**
- Test accuracy: **98.28%**

---

## Run it

### In Colab (one click)
Use the badge above. The notebook contains all steps end-to-end.

### Local
```bash
pip install -r requirements.txt
# then open and run mnist_project.ipynb (Jupyter/VS Code)
```
