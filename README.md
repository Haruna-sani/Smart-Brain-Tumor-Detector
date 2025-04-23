# Brain Tumor Classification using Deep Learning (ResNet50V2)

This project presents a deep learning-based approach for the classification of brain tumor images using a pre-trained **ResNet50V2** model. The model was fine-tuned to detect and classify brain tumors into three categories: **glioma**, **meningioma (menin)**, and **pituitary tumor**.

## 📌 Project Overview

Brain tumor classification from MRI scans is a critical task in the medical domain. This project leverages transfer learning using ResNet50V2, a state-of-the-art convolutional neural network architecture, to accurately classify MRI scans into tumor types with high precision and recall.

## 🧠 Classes

The classification task involves the following three tumor classes:
- **Glioma**
- **Meningioma**
- **Pituitary Tumor** (labelled as 'tumor' in the results)

## 🚀 Model Architecture

The core of this model is built on **ResNet50V2** from `tensorflow.keras.applications`. Key modifications and layers added include:

- **GlobalAveragePooling2D**
- **BatchNormalization**
- **Dense layers with ReLU activation**
- **Dropout for regularization**
- **Final Dense softmax layer for classification**

### Key Callbacks Used:
- **ReduceLROnPlateau**
- **EarlyStopping**

## 📊 Model Performance

The model achieved impressive results on the test set:

### 📈 Classification Report

```
              precision    recall  f1-score   support

      glioma       0.98      0.98      0.98       401
       menin       0.97      0.91      0.94       401
       tumor       0.93      0.99      0.96       410

    accuracy                           0.96      1212
   macro avg       0.96      0.96      0.96      1212
weighted avg       0.96      0.96      0.96      1212
```

Overall accuracy: **96%**

The model demonstrates robust and balanced performance across all classes, with strong generalization capabilities on unseen MRI data.

## 🧰 Libraries Used

The following Python libraries and frameworks were used in this project:

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import tensorflow as tf
from tqdm.keras import TqdmCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
```

## 🏁 Conclusion

This project showcases how transfer learning using ResNet50V2 can be effectively applied to medical image classification problems. With further tuning and more data, this model has potential for clinical support in tumor diagnosis.

## 📌 Future Work

- Incorporate more diverse datasets
- Apply advanced augmentation techniques
- Deploy model with a front-end for medical personnel use

---

Data source: https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset/data

```
