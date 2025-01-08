import numpy as np
import joblib

# Get the feature names (vocabulary)
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Load the vectorizer
feature_names = vectorizer.get_feature_names_out()

# Get the SVM coefficients (weights for each feature)
svm_model = joblib.load("svm_model.pkl")
svm_weights = svm_model.coef_.toarray()

# Find the most important features for each class
def get_top_features(weights, feature_names, top_n=2):
    """
    Get the most influential features for the positive and negative classes.

    Args:
        weights (np.array): Coefficients from the SVM model.
        feature_names (list): List of feature names (words).
        top_n (int): Number of top features to retrieve.

    Returns:
        dict: Dictionary with top positive and negative features.
    """
    top_positive_indices = np.argsort(weights[0])[-top_n:]  # Top N positive
    top_negative_indices = np.argsort(weights[0])[:top_n]  # Top N negative

    top_positive_features = [(feature_names[i], weights[0][i]) for i in reversed(top_positive_indices)]
    top_negative_features = [(feature_names[i], weights[0][i]) for i in top_negative_indices]

    return {
        "positive": top_positive_features,
        "negative": top_negative_features
    }

# Get top 10 influential features for each class
top_features = get_top_features(svm_weights, feature_names, top_n=20)

print("Top features for label 1 (woman):")
for word, weight in top_features["positive"]:
    print(f"{word}: {weight:.4f}")

print("\nTop features for label 0 (man):")
for word, weight in top_features["negative"]:
    print(f"{word}: {weight:.4f}")
