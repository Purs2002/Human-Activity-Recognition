# Human Activity Recognition Using Machine Learning

## Project Overview

Human Activity Recognition (HAR) is a machine learning-based approach used to identify and classify human physical activities from collected sensor data.

This project implements a Human Activity Recognition system using supervised machine learning algorithms. It uses the HAR dataset containing training and testing samples to classify six different human activities:

1. Walking
2. Walking Upstairs
3. Walking Downstairs
4. Sitting
5. Standing
6. Laying

Two machine learning algorithms, **Random Forest Classifier** and **Support Vector Machine (SVM)**, are implemented and evaluated to compare their classification performance.

The objective is to investigate how effectively traditional machine learning techniques can recognize different human activities from extracted input features.

## Objectives

- Develop a machine learning-based Human Activity Recognition system.
- Process and analyze the HAR dataset.
- Train supervised classification models.
- Implement Random Forest and Support Vector Machine algorithms.
- Predict different categories of human physical activities.
- Evaluate classification performance using accuracy, classification reports, and confusion matrices.
- Compare the performance of the implemented machine learning models.

## Dataset

**Dataset:** Human Activity Recognition (HAR) Dataset

The dataset is loaded from Kaggle and consists of two separate files:

- `train.csv` – Training dataset.
- `test.csv` – Testing dataset.

The last column of each dataset is treated as the target label, while all preceding columns are used as input features.

### Activity Classes

| Class Label | Human Activity |
|---|---|
| 1 | Walking |
| 2 | Walking Upstairs |
| 3 | Walking Downstairs |
| 4 | Sitting |
| 5 | Standing |
| 6 | Laying |

## Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Numerical computations |
| Pandas | Dataset loading and manipulation |
| Matplotlib | Data visualization |
| Seaborn | Confusion matrix visualization |
| Scikit-learn | Machine learning algorithms and evaluation |

## Methodology

### 1. Dataset Loading

The training and testing datasets are loaded using Pandas.

```python
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
```

The dataset is divided into input features and target labels.

- X_train: Training features
- y_train: Training labels
- X_test: Testing features
- y_test: Testing labels

### 2. Random Forest Classification

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve classification performance.

Implementation details:

- Algorithm: Random Forest Classifier
- Number of estimators: 100
- Training: Uses the training dataset.
- Prediction: Classifies activities in the testing dataset.

The model predicts one of the six activity classes.

### 3. Support Vector Machine (SVM)

Support Vector Machine is a supervised machine learning algorithm that identifies an optimal decision boundary to separate different classes.

Implementation details:

- Algorithm: Support Vector Classifier (SVC)
- Default Scikit-learn configuration.
- Training: Uses the training dataset.
- Prediction: Classifies human activities using the learned decision boundaries.

### 4. Model Evaluation

Both models are evaluated using the following performance metrics:

**Accuracy Score**

Measures the proportion of correctly classified activity samples.

**Classification Report**

Provides:
- Precision
- Recall
- F1-score
- Support

**Confusion Matrix**

Visualizes the relationship between actual and predicted activity classes and helps identify misclassification patterns.

## System Workflow

```text
       HAR Dataset
             |
             v
      Dataset Loading
         (Pandas)
             |
             v
     Feature and Label
        Separation
             |
             v
      Training Dataset
             |
       +-----+-----+
       |           |
       v           v
   Random Forest   SVM
       |           |
       v           v
    Predictions  Predictions
       |           |
       +-----+-----+
             |
             v
      Model Evaluation
             |
       +-----+------+
       |            |
       v            v
    Accuracy    Classification
                   Report
             |
             v
       Confusion Matrix
             |
             v
     Performance Comparison
```

## Installation

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Project Execution

### Step 1: Download the Dataset

Download the Human Activity Recognition dataset from Kaggle and ensure that `train.csv` and `test.csv` are available.

### Step 2: Configure Dataset Paths

Update the dataset paths according to your local environment.

```python
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
```

### Step 3: Execute the Notebook

Run the notebook in Google Colab, Jupyter Notebook, or Kaggle.

### Step 4: Train Models

Train the Random Forest and SVM classifiers using the training dataset.

### Step 5: Evaluate Performance

Generate predictions and evaluate both models using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Step 6: Compare Predictions

Compare the predictions generated by both models against the actual activity labels.

## Key Features

- Six-class human activity classification.
- Supervised machine learning implementation.
- Random Forest-based classification.
- SVM-based classification.
- Independent training and testing datasets.
- Classification performance evaluation.
- Confusion matrix visualization.
- Comparison of two machine learning algorithms.

## Applications

- Smartphone-based activity monitoring.
- Wearable fitness tracking systems.
- Human movement analysis.
- Healthcare activity monitoring.
- Smart home automation.
- Context-aware computing.
- Physical activity recognition.

## Future Improvements

- Apply feature scaling and compare SVM performance.
- Perform hyperparameter optimization.
- Compare additional algorithms such as KNN, Decision Trees, and XGBoost.
- Perform feature importance analysis.
- Investigate dimensionality reduction using PCA.
- Analyze class-wise performance.
- Explore deep learning architectures such as CNN and LSTM.
- Develop a real-time activity recognition application using sensor inputs.

## Author

**Project:** Human Activity Recognition Using Machine Learning

**Domain:** Machine Learning | Pattern Recognition | Data Analytics

**Algorithms:** Random Forest | Support Vector Machine (SVM)

**Tools:** Python | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn

---

*This project demonstrates the application of supervised machine learning algorithms for multi-class human activity classification and performance evaluation.*
