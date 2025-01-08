"""
Script: TF-IDF Computation and Vocabulary Extraction
Author: Boris, Dani
Date: 24/12/24

Description:
This script preprocesses a dataset of text posts, computes TF-IDF scores for each term in each document, and saves the results (sparse matrix and vocabulary) to CSV files. The preprocessing steps include lowercasing, punctuation removal, stemming, word length filtering, and reducing repeated characters.

Key Features:
- Preprocesses up to 50,000 text posts efficiently using pandas.
- Uses sklearn's TfidfVectorizer for TF-IDF computation.
- Outputs:
  1. A sparse CSV file with TF-IDF scores for each document and term.
  2. A vocabulary CSV file mapping words to indices.

Notes:
- Change the file path to your local path
- The `post` column in the input CSV must contain the text data.
- Works with altered gender csv file where all of the information is in 1 column separated with commas (we've done that manually in excel :D )
- Takes about 1-2 minutes to run
"""


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import string
from scipy.sparse import save_npz, load_npz
import re
import joblib
import numpy as np

def create_document_labels_file(input_file, output_file, max_rows=35000):
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

def preprocess_text(text):
    """
    Preprocess a single text post by:
    1. Convert to lowercase
    2. Remove punctuation
    3. Split into words
    4. Discard words that contain characters outside of [a-z0-9]
    5. Naively normalize words to a base form:
       - Remove 'ing' ending if present
       - Remove 'ed' ending if present
       - Remove trailing 's' if present (for plural forms)
    6. Remove words longer than 25 characters
    7. Reduce occurrences of 3+ consecutive same letters to 2
    8. Return a cleaned list of words

    Examples:
    - "playing" -> "play" (remove 'ing')
    - "kings" -> "king" (remove trailing 's')
    - "Painted" -> "paint" (remove 'ed')
    - "Haaaaappy" -> "haappy" (reduce repeated letters)

    Limitation: words ending with s like boss, however in our program the word bos will be connected with the real word boss so there wont be much of a negative.
    """

    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation by translating them to empty
    # Using str.translate is efficient
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 7. Reduce occurrences of 3+ consecutive same letters to 2
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # 3. Split into words
    words = text.split()

    cleaned_words = []
    for w in words:
        # 4. Check if word contains only [a-z0-9], discard otherwise
        if not re.match('^[a-z0-9]+$', w):
            continue

        # 5. Normalize word endings
        # Remove 'ing' at the end
        if w.endswith('ing') and len(w) > 3:
            w = w[:-3]

        # Remove 'ed' at the end
        elif w.endswith('ed') and len(w) > 2:
            w = w[:-2]

        # Remove trailing 's' to handle simple plural
        # Ensure the word isn't just 's'
        elif w.endswith('s') and len(w) > 1:
            w = w[:-1]

        # 6. Remove words longer than 25 characters
        if len(w) <= 25:
            cleaned_words.append(w)

    return cleaned_words

def load_and_preprocess(file_path, max_rows=35000):
    """
    Loads the CSV using pandas for speed and applies preprocessing.
    This function:
    - Reads the 'post' column from the CSV
    - Applies the preprocess_text function to each post
    - Joins the cleaned words back into a string (so TfidfVectorizer can handle it)
    """
    df = pd.read_csv(file_path, usecols=['post'], nrows=max_rows, encoding='utf-8')
    df['post'] = df['post'].astype(str).apply(lambda x: ' '.join(preprocess_text(x)))
    return df['post'].tolist()

def main():
    input_file = "shuffled_file_gender_neutral.csv"
    output_file = "tf_idf_sparse.csv"
    vocab_file = "vocabulary.csv"
    label_file = "document_labels.csv"
    #input_file = "shuffled_file.csv"

    #uncomment this to create the shuffled file first
    # df = pd.read_csv(input_file)

    # # Shuffle the rows randomly
    # df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # # Save the shuffled DataFrame to a new CSV file
    # shuffled_file_path = "shuffled_file.csv"  # Replace with the desired output file name
    # df_shuffled.to_csv(shuffled_file_path, index=False)
    # print(f"Shuffled data saved to {shuffled_file_path}")
    # return

    print("Loading and preprocessing data...")
    documents = load_and_preprocess(input_file, max_rows=35000)

    print("Fitting TfidfVectorizer...")
    # Use TfidfVectorizer to handle tokenization, vocabulary, and TF-IDF
    # We already processed the text, so we can trust the whitespace tokenization
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        lowercase=False,
        dtype=np.float64,  # Use NumPy float type
        token_pattern=None  # Explicitly disable token_pattern
    )
    tf_idf_matrix = vectorizer.fit_transform(documents)

    print("Saving TF-IDF scores and vocabulary...")

    tf_idf_npz_file = "tf_idf_sparse_matrix.npz"
    save_npz(tf_idf_npz_file, tf_idf_matrix)
    print(f"Sparse TF-IDF matrix saved to {tf_idf_npz_file}")

    print("Saving TfidfVectorizer...")
    vectorizer_file = "tfidf_vectorizer.pkl"
    joblib.dump(vectorizer, vectorizer_file)
    print(f"TfidfVectorizer saved to {vectorizer_file}")

    # Create inverse vocabulary mapping
    '''
    inv_vocab = {idx: word for word, idx in vectorizer.vocabulary_.items()}

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("row,word,score\n")
        coo = tf_idf_matrix.tocoo()
        for row, col, score in zip(coo.row, coo.col, coo.data):
            f.write(f"{row},{inv_vocab[col]},{score:.5f}\n")
    '''
    # Save the vocabulary
    with open(vocab_file, 'w', encoding='utf-8') as f:
        f.write("word,index\n")
        for word, index in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1]):
            f.write(f"{word},{index}\n")

    print(f"Sparse TF-IDF scores saved to {output_file}")
    print(f"Vocabulary saved to {vocab_file}")

    # Save document labels
    create_document_labels_file(input_file, label_file, max_rows=35000)

if __name__ == "__main__":
    main()
