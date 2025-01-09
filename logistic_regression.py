"""
Script: Logistic Regression on a preprocessed dataset
Author: Beloslava Malakova
Date: 30/12/2024, modified 08/01/2025

Description: Applying Logistic regression with L1, L2 and elastic net regularization. Applied is k-fold cross-validation. Accuracy is measured through accuracy, precision, recall and F1 score.

Key Features:
- Evaluates different regularization techniques.
- Metrics: Accuracy, Precision, Recall, F1-score.
- 3-fold cross-validation for robust evaluation.

Notes:
- Incorporated timer.
- When it has to be run for the raw data simply change the datasets in tf_idf_data =, and y.
"""

import time
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support



# labels = pd.read_csv("gender_shuffled_document_labels.csv")
# nan_rows = labels[labels['female'].isna()]
# print("Rows with NaN values:")
# print(nan_rows)
# labels = labels.dropna(subset=['female'])

# Load and transform the TF-IDF data
print("Loading and transforming TF-IDF data...")
tf_idf_data = pd.read_csv("gender_shuffled_tf_idf_sparse.csv")
#tf_idf_data = pd.read_csv("tf_idf_sparse_raw.csv") #works for this
#tf_idf_data['row'] = tf_idf_data['row'].astype(int)  # Document indices
tf_idf_data['document'] = tf_idf_data['document'].astype(int)  # Document indices CHANGE BACK TO ROW
tf_idf_data['word'] = tf_idf_data['word'].astype(int)  # Feature indices
tf_idf_data['score'] = tf_idf_data['score'].astype(float)  # TF-IDF scores

# Get the number of documents and features
#n_documents = tf_idf_data["row"].max() + 1
n_documents = tf_idf_data["document"].max() + 1

n_features = tf_idf_data["word"].max() + 1

# Create a sparse matrix at the document level
#X = coo_matrix(
#    (tf_idf_data['score'], (tf_idf_data['row'], tf_idf_data['word'])), #change back to row
#    shape=(n_documents, n_features)
X = coo_matrix(
    (tf_idf_data['score'], (tf_idf_data['document'], tf_idf_data['word'])), #change back to row
    shape=(n_documents, n_features)
).tocsr()

print(f"Feature matrix shape: {X.shape}")

# Load labels
print("Loading document labels...")
# y = pd.read_csv("document_labels_raw.csv")["female"] # works for this
y = pd.read_csv("gender_shuffled_document_labels.csv")["female"] # shuffled data

# if pd.isna(y).any():
#     raise ValueError("Target labels contain NaN values. Please preprocess your data correctly.")

assert X.shape[0] == len(y), "Mismatch between feature matrix and labels!"

nan_mask = pd.isna(y)  # Identify rows where y is NaN
if nan_mask.any():
    print(f"Found {nan_mask.sum()} NaN values in labels. Removing them...")
    y = y[~nan_mask]  # Keep only non-NaN values in y
    X = X[~nan_mask]  # Remove corresponding rows from X

# Initialize 3-fold cross-validation
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
    start_time = time.time()
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

    # End timer after cross-validation
    end_time = time.time()
    total_run_time = end_time - start_time  # total time in seconds

    # Average metrics across folds
    results[model_name] = {
        metric: sum(values) / len(values) for metric, values in fold_metrics.items()
    }
    results[model_name]["Runtime"] = total_run_time

    print(f"Results for {model_name}:")
    for metric, value in results[model_name].items():
        if metric != "Runtime":
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value:.2f} seconds")

# final results summary
print("\nFinal Comparison Summary:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        if metric == "Runtime":
            print(f"  {metric}: {value:.2f} seconds")
        else:
            print(f"  {metric}: {value:.4f}")
