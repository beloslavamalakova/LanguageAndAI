"""
Script: Logistic Regression on a preprocessed dataset
Author: Beloslava Malakova
Date: 30/12/2024

Description: Applying Logistic regression with L1, L2 and elastic net regularization. Applied is k-fold cross-validation. Accuracy is measured through accuracy, precision, recall and F1 score.

Key Features:
- Evaluates different regularization techniques.
- Metrics: Accuracy, Precision, Recall, F1-score.
- 3-fold cross-validation for robust evaluation.

Notes:
- Does not work with the current TF-IDF matrix, tested however on a ready and simple dataset, where it outputs the results correctly!
"""

import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Step 1: Load and transform the TF-IDF data
print("Loading and transforming TF-IDF data...")
tf_idf_data = pd.read_csv("synthetic_tf_idf_sparse.csv")
tf_idf_data['row'] = tf_idf_data['row'].astype(int)  # Document indices
tf_idf_data['word'] = tf_idf_data['word'].astype(int)  # Feature indices
tf_idf_data['score'] = tf_idf_data['score'].astype(float)  # TF-IDF scores

# Get the number of documents and features
n_documents = tf_idf_data["row"].max() + 1
n_features = tf_idf_data["word"].max() + 1

# Create a sparse matrix at the document level
X = coo_matrix(
    (tf_idf_data['score'], (tf_idf_data['row'], tf_idf_data['word'])),
    shape=(n_documents, n_features)
).tocsr()

print(f"Feature matrix shape: {X.shape}")

# Load labels
print("Loading document labels...")
y = pd.read_csv("synthetic_document_labels.csv")["female"]
assert X.shape[0] == len(y), "Mismatch between feature matrix and labels!"

# Step 3: Initialize 3-fold cross-validation
print("Setting up 3-fold cross-validation...")
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Defining the 4 models
models = {
    "Simple Logistic Regression": LogisticRegression(penalty=None, solver='lbfgs', max_iter=500),
    "Lasso (L1)": LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=500),
    "Ridge (L2)": LogisticRegression(penalty='l2', solver='liblinear', C=1.0, max_iter=500),
    "Elastic Net": LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, max_iter=500)
}

# cross-validation and comparison models
results = {}
for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")
    fold_metrics = {"Accuracy": [], "Precision": [], "Recall": [], "F1 Score": []}

    for train_index, test_index in kf.split(X, y):
        # Splitting data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

        # Store metrics
        fold_metrics["Accuracy"].append(accuracy)
        fold_metrics["Precision"].append(precision)
        fold_metrics["Recall"].append(recall)
        fold_metrics["F1 Score"].append(f1)

    # Average metrics across folds
    results[model_name] = {metric: sum(values) / len(values) for metric, values in fold_metrics.items()}
    print(f"Results for {model_name}:")
    for metric, value in results[model_name].items():
        print(f"  {metric}: {value:.4f}")

# final results summary
print("\nFinal Comparison Summary:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
