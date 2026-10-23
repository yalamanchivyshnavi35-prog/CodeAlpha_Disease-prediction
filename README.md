Disease Prediction from Medical Data
A machine learning project that compares classification algorithms on structured medical data to predict whether a breast tumor record is labeled malignant or benign.
Disclaimer: This project is for educational purposes only. It is not a medical device and must not be used to diagnose, screen, or make treatment decisions.

Project Objective
Build and evaluate classification models that learn from medical features and predict a target category. This implementation uses the Breast Cancer Wisconsin Diagnostic dataset included with scikit-learn.
Dataset
- Dataset: Breast Cancer Wisconsin Diagnostic
- Records: 569
- Features: 30 numeric features
- Target classes: Malignant and benign
- Split: 80% training data and 20% testing data
- Dataset access: Loaded through sklearn.datasets.load_breast_cancer()
The features describe characteristics calculated from digitized images of breast mass samples. This version does not use patient symptoms or blood-test results.
Algorithms Compared
1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest
4. XGBoost
Feature scaling is used in the Logistic Regression and SVM pipelines. Model performance is evaluated with accuracy, precision, recall, F1-score, and a confusion matrix.
Results
Results from the completed run:
Model	Test accuracy
Logistic Regression	98.25%
SVM	98.25%
Random Forest	95.61%
XGBoost	94.74%


Logistic Regression and SVM tied for the highest accuracy on this particular train/test split. These results may change with a different split, configuration, or dataset.
Technologies Used
- Python
- NumPy
- pandas
- scikit-learn
- XGBoost
