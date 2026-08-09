# Machine Learning Assignment 2 — Classification Model Comparison App

## 📌 Project Overview

This project builds, evaluates, and deploys an interactive machine learning application that compares multiple classification algorithms on the same dataset.

The task is **binary classification of breast tumors as malignant or benign** using features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

The application provides an interactive way to compare different classification algorithms based on:

* Accuracy
* AUC
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)
* Confusion Matrix
* Classification Report

The project demonstrates how different machine learning algorithms trade off **accuracy, interpretability, and robustness** on a real-world clinical dataset.

---

# 1. Problem Statement

The objective of this assignment is to develop an interactive machine learning application that:

1. Uses a real-world classification dataset.
2. Trains multiple classification algorithms.
3. Evaluates all models on the same held-out test dataset.
4. Compares model performance using multiple evaluation metrics.
5. Provides an interactive Streamlit interface for model selection and evaluation.
6. Allows users to upload test data and evaluate the trained models.

The selected problem is **breast tumor classification**, where the objective is to classify tumors as:

* **Malignant**
* **Benign**

---

# 2. Dataset Description

### Breast Cancer Wisconsin (Diagnostic) Dataset

**Dataset Source:** UCI Machine Learning Repository

The dataset is also available through:

```python
sklearn.datasets.load_breast_cancer
```

### Dataset Characteristics

| Property            | Value                 |
| ------------------- | --------------------- |
| Number of Instances | 569                   |
| Number of Features  | 30 numeric features   |
| Target Variable     | Binary Classification |
| Malignant Class     | 212                   |
| Benign Class        | 357                   |
| Training Data       | 75%                   |
| Test Data           | 25%                   |
| Test Samples        | 143                   |
| Random State        | 42                    |

The dataset satisfies the assignment requirements of having at least **500 instances** and **12 or more features**.

### Feature Examples

The 30 features describe characteristics of the cell nuclei present in the digitized FNA images, including:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Concave points
* Symmetry
* Fractal dimension

These measurements are provided as:

* Mean values
* Standard error values
* Worst-case values

### Data Preprocessing

The dataset was split into training and testing sets using a **75:25 stratified split**.

```python
train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42
)
```

Features were standardized using `StandardScaler` before training models that are sensitive to feature scale, particularly:

* Logistic Regression
* k-Nearest Neighbors (kNN)

---

# 3. GitHub Repository

**GitHub Repository:**
[PASTE YOUR GITHUB REPOSITORY LINK HERE]

For example:

```text
https://github.com/RajNarayanDeo/Machine-Learning
```

---

# 4. Project Structure

```text
project-folder/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── model/
    ├── train_models.py
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest_ensemble.pkl
    ├── scaler.pkl
    ├── feature_names.json
    └── metrics.json
```

### File Description

| File                         | Description                          |
| ---------------------------- | ------------------------------------ |
| `app.py`                     | Streamlit application                |
| `requirements.txt`           | Python dependencies                  |
| `README.md`                  | Project documentation                |
| `test_data.csv`              | Test dataset used by the application |
| `train_models.py`            | Model training and evaluation script |
| `logistic_regression.pkl`    | Trained Logistic Regression model    |
| `decision_tree.pkl`          | Trained Decision Tree model          |
| `knn.pkl`                    | Trained kNN model                    |
| `naive_bayes.pkl`            | Trained Gaussian Naive Bayes model   |
| `random_forest_ensemble.pkl` | Trained Random Forest model          |
| `scaler.pkl`                 | Feature scaling object               |
| `feature_names.json`         | Feature names used by the models     |
| `metrics.json`               | Stored model evaluation metrics      |

---

# 5. Machine Learning Models

Five classification algorithms were trained using the same training and testing datasets.

### Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

### Model Configuration

| Model                | Configuration                |
| -------------------- | ---------------------------- |
| Logistic Regression  | Standard Logistic Regression |
| Decision Tree        | `max_depth=5`                |
| kNN                  | `k=7`                        |
| Gaussian Naive Bayes | Default Gaussian NB          |
| Random Forest        | 200 trees                    |

---

# 6. Model Performance Comparison

All models were evaluated on the same **143 held-out test samples**.

| ML Model                |   Accuracy |        AUC |  Precision |     Recall |         F1 |        MCC |
| ----------------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| **Logistic Regression** | **0.9860** | **0.9977** | **0.9889** | **0.9889** | **0.9889** | **0.9700** |
| Decision Tree           |     0.9371 |     0.9186 |     0.9551 |     0.9444 |     0.9497 |     0.8657 |
| kNN                     |     0.9790 |     0.9923 |     0.9677 | **1.0000** |     0.9836 |     0.9555 |
| Naive Bayes             |     0.9371 |     0.9878 |     0.9355 |     0.9667 |     0.9508 |     0.8644 |
| Random Forest           |     0.9580 |     0.9950 |     0.9565 |     0.9778 |     0.9670 |     0.9098 |

> These metrics can be reproduced by running `python model/train_models.py`.

---

# 7. Model Performance Analysis

## Logistic Regression

Logistic Regression achieved the **highest overall performance** among the five models.

### Key results

* Accuracy: **98.60%**
* AUC: **99.77%**
* F1 Score: **98.89%**
* MCC: **0.9700**

The dataset appears to be close to linearly separable after feature scaling, making Logistic Regression highly effective.

Another advantage is that Logistic Regression is relatively simple and interpretable compared with tree ensembles.

**Conclusion:** Best overall model for this dataset.

---

## Decision Tree

The Decision Tree achieved:

* Accuracy: **93.71%**
* AUC: **91.86%**
* F1 Score: **94.97%**
* MCC: **0.8657**

The tree was limited to a maximum depth of 5 to reduce overfitting.

However, a single decision tree can be sensitive to individual splits in the data, which resulted in more classification errors compared with the other high-performing models.

**Conclusion:** Easy to interpret but weaker predictive performance for this dataset.

---

## k-Nearest Neighbors (kNN)

kNN achieved:

* Accuracy: **97.90%**
* AUC: **99.23%**
* Recall: **100%**
* F1 Score: **98.36%**
* MCC: **0.9555**

The perfect recall means that the model correctly identified all samples belonging to the evaluated positive class.

Because the features were standardized, Euclidean distance becomes more meaningful across the different feature scales.

**Conclusion:** Excellent performer and a strong runner-up to Logistic Regression.

---

## Gaussian Naive Bayes

Naive Bayes achieved:

* Accuracy: **93.71%**
* AUC: **98.78%**
* F1 Score: **95.08%**
* MCC: **0.8644**

Although the model achieved a high AUC, its accuracy and MCC were lower than the top-performing models.

One possible reason is the Gaussian Naive Bayes assumption that features are conditionally independent.

Many features in this dataset are strongly related because they represent different measurements of similar underlying characteristics.

**Conclusion:** Simple and computationally efficient, but its independence assumption limits performance on this dataset.

---

## Random Forest

Random Forest achieved:

* Accuracy: **95.80%**
* AUC: **99.50%**
* F1 Score: **96.70%**
* MCC: **0.9098**

The model uses an ensemble of **200 decision trees**.

Compared with a single Decision Tree, Random Forest reduces variance by combining predictions from multiple trees.

It achieved excellent AUC and strong overall performance, although it was slightly behind Logistic Regression and kNN.

**Conclusion:** Strong and robust model, but less interpretable than Logistic Regression.

---

# 8. Overall Winner

## 🏆 Logistic Regression

For this particular dataset, **Logistic Regression is the overall winner**.

It achieved:

* Highest Accuracy: **98.60%**
* Highest F1 Score: **98.89%**
* Highest MCC: **0.9700**
* Highest AUC: **99.77%**

It also provides an important practical advantage: **interpretability**.

Therefore, Logistic Regression provides the best balance of:

**Performance + Simplicity + Interpretability**

kNN is a very close runner-up because of its **100% recall** and strong AUC.

> **Important:** The "best" model is dataset- and objective-dependent. In a real clinical application, model selection would also require validation on independent data, calibration, clinical evaluation, and careful consideration of the relative cost of false negatives and false positives.

---

# 9. Streamlit Application

The project includes an interactive Streamlit application for comparing the trained classification models.

## Application Features

### 1. CSV Upload

Users can upload:

```text
test_data.csv
```

or another CSV containing the same 30 feature columns and a `target` column.

---

### 2. Model Selection

Users can select one of the following models from a dropdown:

* Logistic Regression
* Decision Tree
* kNN
* Naive Bayes
* Random Forest

---

### 3. Evaluation Metrics

The application calculates and displays:

* Accuracy
* AUC
* Precision
* Recall
* F1 Score
* MCC

---

### 4. Model Comparison

The application displays a comparison table containing the training-time performance of all five models.

---

### 5. Confusion Matrix

A visual confusion matrix is provided to show:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

### 6. Classification Report

The application provides a detailed classification report containing per-class:

* Precision
* Recall
* F1 Score
* Support

---

# 10. How to Run the Project Locally

## Step 1 — Clone the Repository

```bash
git clone https://github.com/RajNarayanDeo/Machine-Learning.git
```

Navigate to the project directory:

```bash
cd Machine-Learning
```

---

## Step 2 — Create a Virtual Environment

### Windows

```powershell
python -m venv venv
```

Activate the environment:

```powershell
venv\Scripts\activate
```

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4 — Retrain Models

If you want to regenerate the trained models and metrics:

```bash
python model/train_models.py
```

This generates/updates:

```text
model/
├── logistic_regression.pkl
├── decision_tree.pkl
├── knn.pkl
├── naive_bayes.pkl
├── random_forest_ensemble.pkl
├── scaler.pkl
├── feature_names.json
└── metrics.json
```

---

## Step 5 — Launch the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 11. Technologies Used

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Python       | Programming language           |
| Scikit-learn | Machine learning models        |
| Pandas       | Data manipulation              |
| NumPy        | Numerical computation          |
| Matplotlib   | Visualization                  |
| Seaborn      | Confusion matrix visualization |
| Streamlit    | Interactive web application    |
| Joblib       | Model serialization            |
| Git          | Version control                |
| GitHub       | Source code repository         |

---

# 12. Evaluation Metrics

### Accuracy

Measures the percentage of correctly classified samples.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how many predicted positive samples are actually positive.

```text
Precision = TP / (TP + FP)
```

### Recall

Measures how many actual positive samples were correctly identified.

```text
Recall = TP / (TP + FN)
```

### F1 Score

Harmonic mean of precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### AUC

Measures the model's ability to distinguish between the two classes across different classification thresholds.

### Matthews Correlation Coefficient (MCC)

MCC provides a balanced measure of classification quality using all four confusion-matrix categories:

* True Positive
* True Negative
* False Positive
* False Negative

---

# 13. Deployment

## Live Streamlit Application

**Live App:**
[PASTE YOUR STREAMLIT COMMUNITY CLOUD APP LINK HERE]

The deployed application allows users to interactively:

1. Upload test data.
2. Select a classification model.
3. Generate predictions.
4. View evaluation metrics.
5. Inspect the confusion matrix.
6. View the classification report.
7. Compare all five models.

---

# 14. Key Findings

The experiment demonstrates that different classification algorithms can produce significantly different results on the same dataset.

### Main observations

* **Logistic Regression** achieved the best overall performance.
* **kNN** achieved perfect recall and was the strongest alternative.
* **Random Forest** provided strong performance and excellent AUC.
* **Decision Tree** performed worse than the ensemble Random Forest.
* **Naive Bayes** was affected by feature dependency assumptions.
* Feature scaling was particularly important for Logistic Regression and kNN.
* Model selection should consider both predictive performance and interpretability.

---

# 15. Conclusion

This project demonstrates a complete machine learning workflow, from dataset preparation and preprocessing to model training, evaluation, comparison, and interactive deployment.

Five classification algorithms were trained and evaluated using the same train/test split. Among the tested models, **Logistic Regression achieved the best overall performance**, with an accuracy of **98.60%** and an AUC of **99.77%**.

The Streamlit application makes it possible to interactively compare the models and understand their strengths and weaknesses through multiple evaluation metrics and visualizations.

The project therefore demonstrates how machine learning models can be systematically evaluated and compared rather than relying on a single performance metric.

---

## 👨‍💻 Author

**Raj Narayan Deo**

Machine Learning

**GitHub:**
https://github.com/RajNarayanDeo/Machine-Learning
