# Machine Learning Assignment 2 — Classification Model Comparison App

## a. Problem Statement

The goal of this assignment is to build, evaluate, and deploy an interactive
machine learning application that compares multiple classification algorithms
on the same dataset. The task chosen is **binary classification of breast
tumors as malignant or benign** based on characteristics computed from a
digitized image of a fine needle aspirate (FNA) of a breast mass. An accurate,
interpretable, and interactive model comparison tool helps illustrate how
different classification algorithms trade off accuracy, interpretability, and
robustness on a real clinical dataset.

## b. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset
**Source:** UCI Machine Learning Repository (also bundled with `scikit-learn`
as `sklearn.datasets.load_breast_cancer`)

| Property | Value |
|---|---|
| Number of instances | 569 (≥ 500 required ✅) |
| Number of features | 30 numeric features (≥ 12 required ✅) |
| Target variable | Binary: 0 = malignant, 1 = benign |
| Class balance | 212 malignant, 357 benign |
| Feature examples | radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension (mean, standard-error, and "worst" value of each) |

The data was split 75% train / 25% test (stratified) using `random_state=42`
for reproducibility. Features were standardized with `StandardScaler` before
training distance-/gradient-based models (Logistic Regression, kNN).

## c. GitHub Repository Link

`[PASTE YOUR GITHUB REPOSITORY LINK HERE AFTER YOU PUSH THIS PROJECT]`

Repository structure:
```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- test_data.csv
│-- model/
    │-- train_models.py
    │-- logistic_regression.pkl
    │-- decision_tree.pkl
    │-- knn.pkl
    │-- naive_bayes.pkl
    │-- random_forest_ensemble.pkl
    │-- scaler.pkl
    │-- feature_names.json
    │-- metrics.json
```

## d. Models Used

Five classification models were trained on identical train/test splits of the
same dataset:

1. Logistic Regression
2. Decision Tree Classifier (max_depth=5)
3. K-Nearest Neighbor Classifier (k=7)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble, 200 trees)

### Comparison Table (on held-out test set, 143 samples)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9860 | 0.9977 | 0.9889 | 0.9889 | 0.9889 | 0.9700 |
| Decision Tree | 0.9371 | 0.9186 | 0.9551 | 0.9444 | 0.9497 | 0.8657 |
| kNN | 0.9790 | 0.9923 | 0.9677 | 1.0000 | 0.9836 | 0.9555 |
| Naive Bayes | 0.9371 | 0.9878 | 0.9355 | 0.9667 | 0.9508 | 0.8644 |
| Random Forest (Ensemble) | 0.9580 | 0.9950 | 0.9565 | 0.9778 | 0.9670 | 0.9098 |

*(These exact numbers are reproduced automatically by running
`python model/train_models.py`, which regenerates `model/metrics.json`.)*

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved the highest accuracy, F1, and MCC of all five models. The dataset's classes are close to linearly separable after scaling, which suits a linear decision boundary well, and the model generalized cleanly with no signs of overfitting. |
| Decision Tree | Weakest performer on accuracy and MCC. A single tree with limited depth captures the coarse structure of the data but is more sensitive to individual splits, leading to a few more misclassifications than the ensemble/linear approaches. |
| kNN | Very strong performer — achieved perfect recall (caught every benign case) and the second-highest AUC. Performed well because scaled Euclidean distance is meaningful for this dataset's continuous, well-separated features. |
| Naive Bayes | Tied with Decision Tree for lowest accuracy/MCC, mainly because the Gaussian independence assumption between the 30 (fairly correlated) features is violated — many of these features are mean/SE/worst versions of the same underlying measurement. |
| Random Forest (Ensemble) | Excellent AUC (second only to Logistic Regression) and solid all-round metrics. Averaging 200 trees reduces the variance problem seen in the single Decision Tree, at the cost of being a "black box" compared to Logistic Regression. |
| **Overall Winner for this dataset** | **Logistic Regression** — best Accuracy, F1, and MCC, and essentially tied for best AUC, while also being the simplest and most interpretable model of the five. kNN is a very close runner-up given its perfect recall. |

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd project-folder

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Retrain models from scratch
python model/train_models.py

# 4. Launch the Streamlit app
streamlit run app.py
```

## Streamlit App Features

- **CSV Upload:** Upload `test_data.csv` (or any CSV with the same 30 feature
  columns + a `target` column) to evaluate models on that data.
- **Model Selection Dropdown:** Choose between Logistic Regression, Decision
  Tree, kNN, Naive Bayes, and Random Forest.
- **Evaluation Metrics Display:** Live Accuracy, AUC, Precision, Recall, F1,
  and MCC computed on the uploaded data, plus a comparison table of all 5
  models' training-time metrics.
- **Confusion Matrix & Classification Report:** Visual confusion matrix heatmap
  and a full per-class classification report.

## Live App Link

`[PASTE YOUR STREAMLIT COMMUNITY CLOUD APP LINK HERE]`
