import joblib

# Load the trained SVM model and vectorizer
svm_model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Load the vectorizer

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
    #print(processed_text)

    # Predict the label
    prediction = model.predict(processed_text)
    return prediction[0]

# Example sentence
example_sentence = "I am a person imo thank you xd"
predicted_label = predict_label(example_sentence, vectorizer, svm_model)
print(f"The predicted label for '{example_sentence}' is: {predicted_label}")
