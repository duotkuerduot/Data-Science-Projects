# 🍅 Tomato Crop Disease Detection in Kenya

## 📌 Problem Statement
Tomato farming in Kenya, a vital agricultural sector, faces significant threats from various diseases. These diseases lead to substantial crop loss, impacting food security and the livelihoods of farmers.  

Accurate and quick identification of tomato diseases is a major challenge, particularly for **small-scale farmers** who may lack access to expert knowledge or diagnostic tools.

---

## ✅ Solution Steps

### 1. Data Preparation and Preprocessing
The project utilizes a dataset of tomato leaf images, structured into **train** and **val** directories. Key steps include:

- **Data Augmentation**: Training images are enhanced with transformations such as random resizing, horizontal flips, and rotations. This improves robustness and reduces overfitting.  
- **Data Splitting**: The train dataset is programmatically split into a new training set and an unseen test set. The `val` directory is used as the dedicated validation set.  
- **Normalization**: All images are normalized using ImageNet mean and standard deviation values for compatibility with pre-trained models.  
- **Data Loaders**: PyTorch `DataLoader` objects are created for training, validation, and test sets to efficiently batch and load data.

---

### 2. Model Architecture and Training
A **deep learning approach** is applied using a pre-trained **ResNet-50** model.

- **Transfer Learning**: The ResNet-50 model, pre-trained on ImageNet, is used as a feature extractor. Its final classification layer is replaced for tomato disease classification.  
- **Dropout Layer**: A dropout layer (`p=0.5`) is added to the final classification head to prevent overfitting.  
- **Training Loop**: The model is trained with **Stochastic Gradient Descent (SGD)** optimizer and **Cross-Entropy Loss**, with a learning rate scheduler to adjust learning dynamically.

---

### 3. Model Evaluation and Visualization
After training, the model is evaluated on the unseen test dataset.

- **Test Accuracy**: Provides an unbiased measure of generalization performance.  
- **Confusion Matrix**: Visualizes correct vs. incorrect predictions per class, highlighting specific diseases the model struggles with.  
- **Performance Plots**: Training/validation loss and accuracy curves are plotted to monitor training progress, detect overfitting, and confirm model convergence.  

---
