# Neural Network Model for Predicting Alphabet Soup Funding Success

## Overview
This project involves developing and optimizing a deep learning model to predict whether an Alphabet Soup-funded organization will be successful based on provided features. The process includes data preprocessing, model compilation, training, evaluation, and multiple trials to improve accuracy.

---

## Data Preprocessing
1. **Dataset Used:** `charity_data.csv`
2. **Target Variable:** `IS_SUCCESSFUL` (1: successful, 0: not successful)
3. **Feature Variables:** 
   - Various categorical and numerical columns representing organization characteristics.
4. **Data Cleaning & Feature Selection:**
   - Dropped unnecessary columns like `EIN` and `NAME`.
   - Categorical variables were encoded using `pd.get_dummies()`.
   - Scaled numerical features using `StandardScaler()` to ensure proper model performance.
   - Data was split into training and testing sets using `train_test_split()`.

---

## Initial Model Training
Before attempting to optimize, an initial deep learning model was built using:
- A **Sequential model** with three hidden layers.
- **ReLU activation** in hidden layers.
- **Sigmoid activation** in the output layer for binary classification.
- **Binary cross-entropy** as the loss function.
- **Adam optimizer** to train the model.

### Initial Model Performance:
- Accuracy: ~72.6%
- Loss: ~0.56
- The model performed decently but did not reach the **75% target accuracy**.

---

## Optimization Attempts
After training the initial model, five different optimization approaches were tested to improve accuracy.

### **First Trial: Adding More Layers & Neurons**
- Increased neurons in hidden layers.
- Added an extra hidden layer.
- Used **ReLU activation** in all layers.
- **Results:** Accuracy remained at ~72.6%.

### **Second Trial: Trying Different Activation Functions**
- Experimented with `tanh` and `selu` activation functions.
- Kept the same structure as the first trial.
- **Results:** Accuracy remained around ~72.5%.

### **Third Trial: Using Leaky ReLU**
- Introduced **Leaky ReLU** to some layers to improve gradient flow.
- **Results:** Accuracy stayed at ~72.6%.

### **Fourth Trial: Adjusting Input Features**
- Dropped low-correlation columns.
- Retained only relevant features.
- **Results:** No significant accuracy improvement (~72.5%).

### **Fifth Trial: Applying Principal Component Analysis (PCA)**
- Used PCA to reduce dimensionality and capture 95% of variance.
- **Results:** Accuracy still hovered around ~72.6%.

---

## Final Evaluation & Recommendations
- **None of the optimization attempts achieved 75% accuracy.**
- **Possible Next Steps:**
  - Try **different model architectures** like Convolutional or Recurrent Neural Networks.
  - Experiment with **hyperparameter tuning** using GridSearchCV or Bayesian Optimization.
  - Introduce **dropout layers** to prevent overfitting.
  - Collect **more relevant data** to improve feature representation.

---

## Summary
Overall, the deep learning model was able to predict funding success with an accuracy of approximately 72.6%. Various optimization methods were tested, including adding layers, changing activation functions, using Leaky ReLU, adjusting input features, and applying PCA. However, none of these trials significantly improved accuracy beyond the initial model.

A more complex model, such as a convolutional neural network (CNN) or a recurrent neural network (RNN), may be necessary to capture deeper patterns in the data. Additionally, feature engineering and hyperparameter tuning could further enhance performance. Future improvements should focus on obtaining a better dataset, refining feature selection, and testing different neural network architectures.

---

## Model Saving
The final trained model was saved in an HDF5 file:
```python
model.save("trained_model.h5")

---

## Attribution and Code Sources
This project was developed as part of a bootcamp program. **Class materials, documentation, and external resources** were used to complete this challenge. The dataset and methodologies integrate:

- **Neural network modeling using TensorFlow and Keras**
- **Data preprocessing using Pandas and NumPy**
- **Feature scaling using Scikit-Learn’s StandardScaler**
- **Dimensionality reduction using PCA from Scikit-Learn**
- **Model evaluation using Scikit-Learn metrics (Accuracy, Loss Evaluation)**
