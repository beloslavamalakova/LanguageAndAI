import joblib

# Load the input file to dynamically determine base names
input_file = "gender_preprocessed_neutral_gender.csv"  # Replace with your input file
base_name = input_file.split('.')[0]  # Extract base name from input file

# Load the trained SVM model and vectorizer
svm_model_file = f"{base_name}_svm_model.pkl"
vectorizer_file = f"{base_name}_tfidf_vectorizer.pkl"

svm_model = joblib.load(svm_model_file)
vectorizer = joblib.load(vectorizer_file)  # Load the vectorizer

# Function to predict the label for a single text
def predict_label(text, vectorizer, model):
    """
    Predict the label of a given text using the trained SVM model.

    Args:
        text (str): The input text to classify.
        vectorizer: The vectorizer used for training.
        model (SVC): The trained SVM model.

    Returns:
        str: The predicted label for the input text.
    """
    # Transform the input text into a TF-IDF vector
    processed_text = vectorizer.transform([text])  # Transform the text into a TF-IDF vector

    # Predict the label
    prediction = model.predict(processed_text)
    return prediction[0]

# Example sentence
example_sentence = "you and it would be coworker"
predicted_label = predict_label(example_sentence, vectorizer, svm_model)
print(f"The predicted label for '{example_sentence}' is: {predicted_label}")
