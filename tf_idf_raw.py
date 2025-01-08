import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import csv


def create_document_labels_file(input_file, output_file, max_rows=50000):
    """
    Reads the 'female' column from the CSV, along with the row index,
    and saves them into a new CSV file (document_labels_raw.csv).

    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output CSV file.
        max_rows (int): Maximum number of rows to process.
    """
    df = pd.read_csv(input_file, usecols=['female'], nrows=max_rows, encoding='utf-8')
    df.reset_index(inplace=True)  # resets the index → becomes "document_number"

    # Rename columns for clarity
    df.columns = ['document_number', 'female']

    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Document labels saved to {output_file}")

def main():
    input_file     = "gender.csv"              # CSV with 'post' and 'female'
    output_file    = "tf_idf_sparse_raw.csv"   # Will store row, word(index), score
    vocab_file     = "vocabulary_raw.csv"      # Will store word -> numeric index
    label_file     = "document_labels_raw.csv" # Will store document_number, female
    max_rows       = 50000

    print("Loading raw data (unprocessed)...")
    df = pd.read_csv(input_file, usecols=['post'], nrows=max_rows, encoding='utf-8')
    # Replace NaN with "" to avoid errors
    df['post'] = df['post'].fillna("").astype(str)

    # Convert column to a list of raw text
    documents = df['post'].tolist()

    print("Fitting TfidfVectorizer on raw text data...")
    vectorizer = TfidfVectorizer(
        dtype=np.float64
        # If you prefer to keep default tokenization/punctuation, no token_pattern is needed
        # If you do want simpler word-based splits, you could add: token_pattern=r"(?u)\b\w+\b"
    )
    tf_idf_matrix = vectorizer.fit_transform(documents)

    # Inverse vocab mapping: feature_index -> token
    inv_vocab = {idx: word for word, idx in vectorizer.vocabulary_.items()}

    print("Saving TF-IDF scores (with numeric 'word' indices) to CSV...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("row,word,score\n")
        coo = tf_idf_matrix.tocoo()
        for row, col, score in zip(coo.row, coo.col, coo.data):
            # 'row' = document index, 'col' = feature index
            f.write(f"{row},{col},{score:.5f}\n")

    print(f"Sparse TF-IDF scores saved to {output_file}")

    print("Saving vocabulary mapping to CSV...")
    with open(vocab_file, 'w', encoding='utf-8') as f:
        f.write("word,index\n")
        for word, index in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1]):
            f.write(f"{word},{index}\n")

    print(f"Vocabulary saved to {vocab_file}")

    # Save document labels (female)
    create_document_labels_file(input_file, label_file, max_rows=max_rows)

if __name__ == "__main__":
    main()
