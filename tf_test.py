"""
Script: TF-IDF Computation and Vocabulary Extraction (Fixed)
Author: Boris, Dani
Date: 24/12/24

Description:
This script preprocesses a dataset of text posts, computes TF-IDF scores for each term in each document,
and saves the results (sparse matrix and vocabulary) to CSV files.
The key fix is that in the final "tf_idf_sparse.csv," we now store the **integer column index**
instead of the word string, so that logistic_regression.py can do 'astype(int)' without error.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import string
import re
import numpy as np

def create_document_labels_file(input_file, output_file, max_rows=50000):
    """
    Reads the original dataset, extracts the row index and 'female' column,
    and saves them into a new CSV file.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
        max_rows (int): Maximum number of rows to process.
    """
    df = pd.read_csv(input_file, usecols=['female'], nrows=max_rows, encoding='utf-8')
    df.reset_index(inplace=True)  # Reset the index to use it as document numbers

    # Rename columns for clarity
    df.columns = ['document_number', 'female']

    # Save to a new CSV file
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Document labels saved to {output_file}")

def preprocess_text(text):
    """
    Preprocess a single text post by:
    1. Convert to lowercase
    2. Remove punctuation
    3. Split into words
    4. Discard words that contain characters outside of [a-z0-9]
    5. Naively normalize words to a base form (remove ing, ed, s)
    6. Remove words longer than 25 characters
    7. Reduce occurrences of 3+ consecutive same letters to 2
    8. Return a cleaned list of words
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation by translating them to empty
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 7. Reduce occurrences of 3+ consecutive same letters to 2
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # 3. Split into words
    words = text.split()

    cleaned_words = []
    for w in words:
        # 4. Only keep [a-z0-9]
        if not re.match('^[a-z0-9]+$', w):
            continue

        # 5. Remove endings
        if w.endswith('ing') and len(w) > 3:
            w = w[:-3]
        elif w.endswith('ed') and len(w) > 2:
            w = w[:-2]
        elif w.endswith('s') and len(w) > 1:
            w = w[:-1]

        # 6. Remove words longer than 25 characters
        if len(w) <= 25:
            cleaned_words.append(w)

    return cleaned_words

def load_and_preprocess(file_path, max_rows=50000):
    """
    Loads the CSV using pandas for speed and applies preprocessing.
    - Reads the 'post' column
    - Applies preprocess_text() to each post
    - Joins words back into a space-delimited string
    """
    df = pd.read_csv(file_path, usecols=['post'], nrows=max_rows, encoding='utf-8')
    df['post'] = df['post'].astype(str).apply(lambda x: ' '.join(preprocess_text(x)))
    return df['post'].tolist()

def main():
    input_file  = "gender.csv"
    output_file = "tf_idf_sparse.csv"
    vocab_file  = "vocabulary.csv"
    label_file  = "document_labels.csv"

    print("Loading and preprocessing data...")
    documents = load_and_preprocess(input_file, max_rows=50000)

    print("Fitting TfidfVectorizer...")
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        lowercase=False,
        dtype=np.float64,
        token_pattern=None
    )
    tf_idf_matrix = vectorizer.fit_transform(documents)

    # Create inverse vocabulary mapping (feature_index -> word)
    inv_vocab = {idx: word for word, idx in vectorizer.vocabulary_.items()}

    print("Saving TF-IDF scores (as numeric feature indices) and vocabulary...")

    # ----------------------------------------------------------------
    # ********** IMPORTANT CHANGE HERE **********
    # Instead of writing the actual token into the 'word' column,
    # we write the numeric column index. That way logistic_regression.py
    # can do .astype(int) on "tf_idf_data['word']" successfully.
    # ----------------------------------------------------------------
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("row,word,score\n")
        coo = tf_idf_matrix.tocoo()
        for row, col, score in zip(coo.row, coo.col, coo.data):
            # 'row' = document index, 'col' = feature index
            f.write(f"{row},{col},{score:.5f}\n")

    # Save the *textual* vocabulary separately if needed
    # "word,index"
    with open(vocab_file, 'w', encoding='utf-8') as f:
        f.write("word,index\n")
        for word, index in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1]):
            f.write(f"{word},{index}\n")

    print(f"Sparse TF-IDF scores saved to {output_file}")
    print(f"Vocabulary saved to {vocab_file}")

    # Save document labels
    create_document_labels_file(input_file, label_file, max_rows=50000)

if __name__ == "__main__":
    main()
