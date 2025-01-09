import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import numpy as np
import joblib

def create_document_labels_file(input_file, output_file, max_rows=45000):
    """
    Reads the original dataset, extracts the row index and 'female' column, 
    and saves them into a new CSV file.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
        max_rows (int): Maximum number of rows to process.
    """
    # Load the 'female' column from the CSV, along with the row index
    df = pd.read_csv(input_file, usecols=['female'], nrows=max_rows, encoding='utf-8')
    df.reset_index(inplace=True)  # Reset the index to use it as document numbers
    
    # Rename columns for clarity
    df.columns = ['document_number', 'female']
    
    # Save to a new CSV file
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Document labels saved to {output_file}")

def load_text_data(file_path, max_rows=45000):
    """
    Loads the CSV and extracts the 'post' column for processing.
    
    Args:
        file_path (str): Path to the input CSV file.
        max_rows (int): Maximum number of rows to process.

    Returns:
        list: A list of text data from the 'post' column.
    """
    df = pd.read_csv(file_path, usecols=['post'], nrows=max_rows, encoding='utf-8')
    return df['post'].astype(str).tolist()

def save_sparse_matrix_as_csv(tf_idf_matrix, vectorizer, output_file):
    """
    Saves the TF-IDF sparse matrix to a CSV file with three columns: document, word_index, and tf-idf value.

    Args:
        tf_idf_matrix (scipy.sparse.csr_matrix): The sparse TF-IDF matrix.
        vectorizer (TfidfVectorizer): The fitted TfidfVectorizer object.
        output_file (str): Path to save the CSV file.
    """
    print("Saving sparse matrix as CSV...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("document,word_index,tfidf_value\n")
        coo_matrix = tf_idf_matrix.tocoo()
        for row, col, value in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
            f.write(f"{row},{col},{value}\n")
    print(f"Sparse matrix saved to {output_file}")

def main():
    input_file = "gender_preprocessed_neutral_gender.csv"
    sparse_csv_file = "tf_idf_sparse.csv"
    tf_idf_npz_file = "tf_idf_sparse_matrix.npz"
    vocab_file = "vocabulary.csv"
    label_file = "document_labels.csv"

    print("Loading text data...")
    documents = load_text_data(input_file, max_rows=45000)

    print("Fitting TfidfVectorizer...")
    # Use TfidfVectorizer to handle tokenization, vocabulary, and TF-IDF
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        lowercase=False,
        dtype=np.float64,  # Use NumPy float type
        token_pattern=None  # Explicitly disable token_pattern
    )
    tf_idf_matrix = vectorizer.fit_transform(documents)

    print("Saving TF-IDF scores and vocabulary...")

    # Save the sparse matrix in NPZ format
    save_npz(tf_idf_npz_file, tf_idf_matrix)
    print(f"Sparse TF-IDF matrix saved to {tf_idf_npz_file}")

    print("Saving TfidfVectorizer...")
    vectorizer_file = "tfidf_vectorizer.pkl"
    joblib.dump(vectorizer, vectorizer_file)
    print(f"TfidfVectorizer saved to {vectorizer_file}")

    # Save the vocabulary
    with open(vocab_file, 'w', encoding='utf-8') as f:
        f.write("word,index\n")
        for word, index in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1]):
            f.write(f"{word},{index}\n")

    print(f"Vocabulary saved to {vocab_file}")

    # Save document labels
    create_document_labels_file(input_file, label_file, max_rows=45000)

    # Save sparse matrix as CSV
    save_sparse_matrix_as_csv(tf_idf_matrix, vectorizer, sparse_csv_file)

if __name__ == "__main__":
    main()
