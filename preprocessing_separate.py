def preprocess_initial(text):
    """
    Applies the initial preprocessing steps:
    1. Convert to lowercase
    2. Remove punctuation
    3. Split into words
    4. Discard words containing characters outside of [a-z0-9]
    5. Naively normalize words (removal of 'ing', 'ed', trailing 's')
    6. Remove words longer than 25 characters
    7. Reduce occurrences of 3+ consecutive same letters to 2

    Args:
        text (str): The input text to process.

    Returns:
        str: The processed text.
    """
    import string
    import re

    if not isinstance(text, str):  # Handle non-string inputs
        return ""

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    words = text.split()

    cleaned_words = []
    for w in words:
        if not re.match('^[a-z0-9]+$', w):
            continue

        if w.endswith('ing') and len(w) > 3:
            w = w[:-3]
        elif w.endswith('ed') and len(w) > 2:
            w = w[:-2]
        elif w.endswith('s') and len(w) > 1:
            w = w[:-1]

        if len(w) <= 25:
            cleaned_words.append(w)

    return ' '.join(cleaned_words)

def preprocess_gender_neutral(text, replacement_dict):
    """
    Replaces gendered words in the given text with gender-neutral equivalents.

    Args:
        text (str): The input text to process.
        replacement_dict (dict): Dictionary of patterns and replacements.

    Returns:
        str: The processed text with gendered words replaced.
    """
    import re

    if not isinstance(text, str):  # Handle non-string inputs
        return ""

    for pattern, replacement in replacement_dict.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def shuffle_and_save(input_file, shuffled_file, seed=42):
    """
    Shuffles the input file and saves the shuffled version to a new file.

    Args:
        input_file (str): Path to the input CSV file.
        shuffled_file (str): Path to save the shuffled CSV file.
        seed (int): Seed for reproducible shuffling.

    Returns:
        None
    """
    import pandas as pd

    df = pd.read_csv(input_file, encoding='utf-8')
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_shuffled.to_csv(shuffled_file, index=False)
    print(f"Shuffled file saved to {shuffled_file}")

def preprocess_and_save(input_file, output_file_preprocessed, output_file_gender_corrected, replacement_dict, max_rows=45000):
    """
    Preprocesses text from an input file and creates two versions:
    1. Text after initial preprocessing.
    2. Text after gender-neutral replacements and initial preprocessing.

    Args:
        input_file (str): Path to the input CSV file.
        output_file_preprocessed (str): Path to save the preprocessed output.
        output_file_gender_corrected (str): Path to save the gender-neutral preprocessing output.
        replacement_dict (dict): Dictionary of gendered words and replacements.
        max_rows (int): Maximum number of rows to process.

    Returns:
        None
    """
    import pandas as pd

    df = pd.read_csv(input_file, usecols=['auhtor_ID', 'post', 'female'], nrows=max_rows, encoding='utf-8')

    # Ensure no NaN values in the 'post' column
    df['post'] = df['post'].fillna("")

    # Initial preprocessing
    df['post_preprocessed'] = df['post'].apply(preprocess_initial)

    # Gender-neutral preprocessing
    df['post_gender_corrected'] = df['post_preprocessed'].apply(lambda x: preprocess_gender_neutral(x, replacement_dict))

    # Save outputs
    df[['auhtor_ID', 'post_preprocessed', 'female']].rename(columns={'post_preprocessed': 'post'}).to_csv(output_file_preprocessed, index=False)
    df[['auhtor_ID', 'post_gender_corrected', 'female']].rename(columns={'post_gender_corrected': 'post'}).to_csv(output_file_gender_corrected, index=False)

    print(f"Preprocessed text saved to {output_file_preprocessed}")
    print(f"Gender-neutral corrected text saved to {output_file_gender_corrected}")

def count_words(file_path, text_column='post'):
    """
    Count the total and unique number of words in a specified text column of a CSV file.

    Args:
        file_path (str): Path to the CSV file.
        text_column (str): Name of the column containing text.

    Returns:
        tuple: Total number of words, Number of unique words.
    """
    import pandas as pd

    df = pd.read_csv(file_path, usecols=[text_column], encoding='utf-8')
    all_words = df[text_column].dropna().str.split().explode()
    total_words = all_words.count()
    unique_words = all_words.nunique()
    return total_words, unique_words
if __name__ == "__main__":
    replacement_dict = {
        r"\b(fe)?male(s)?\b": "person",
        r"\b(wo)?m[ae]n\b": "person",
        r"\b(s)?he\b": "they",
        r"\bpolice(wo)?m[ae]n\b": "person",
        r"\bpost(wo)?m[ae]n\b": "person",
        r"\bher\b": "them",
        r"\bhers\b": "their",
        r"\bherself\b": "themselves",
        r"\bgal(s)?\b": "person",
        r"\bqueen(s)?\b": "person",
        r"\bgirl(s)?\b": "person",
        r"\bwife\b": "person",
        r"\bwives\b": "person",
        r"\bm[uo]m(s)?\b": "person",
        r"\bmother(s)?\b": "person",
        r"\blady\b": "person",
        r"\bladies\b": "person",
        r"\baunt(s)?\b": "person",
        r"\bniece(s)?\b": "person",
        r"\bsister(s)?\b": "person",
        r"\bdaughter(s)?\b": "person",
        r"\bdame(s)?\b": "person",
        r"\bempress(es)?\b": "person",
        r"\bgirlfriend(s)?\b": "person",
        r"\bgf(s)?\b": "person",
        r"\bprincess(es)?\b": "person",
        r"\bduchess(es)?\b": "person",
        r"\bwaitress(es)?\b": "person",
        r"\bactress(es)?\b": "person",
        r"\bgoddess(es)?\b": "person",
        r"\bheroine(s)?\b": "person",
        r"\bwitch(es)?\b": "person",
        r"\bstewardess(es)?\b": "person",
        r"\bhim\b": "them",
        r"\bhis\b": "their",
        r"\bhimself\b": "themselves",
        r"\bking(s)?\b": "person",
        r"\bboy(s)?\b": "person",
        r"\bguy(s)?\b": "person",
        r"\bhusband(s)?\b": "person",
        r"\bdad(s)?\b": "person",
        r"\bfather(s)?\b": "person",
        r"\bson(s)?\b": "person",
        r"\bbrother(s)?\b": "person",
        r"\buncle(s)?\b": "person",
        r"\bnephew(s)?\b": "person",
        r"\bemperor(s)?\b": "person",
        r"\blord(s)?\b": "person",
        r"\bboyfriend(s)?\b": "person",
        r"\bbf(s)?\b": "person",
        r"\bprince(s)?\b": "person",
        r"\bduke(s)?\b": "person",
        r"\bknight(s)?\b": "person",
        r"\bwaiter(s)?\b": "person",
        r"\bactor(s)?\b": "person",
        r"\bgod(s)?\b": "person",
        r"\bhero(es)?\b": "person",
        r"\bwizard(s)?\b": "person",
        r"\bsteward(s)?\b": "person",
        r"\bpresident(s)?\b": "person",
        r"\bhost(s)?\b": "person",
    }

    input_file = "gender_working.csv"
    shuffled_file = "gender_shuffled.csv"
    output_file_preprocessed = "gender_preprocessed.csv"
    output_file_gender_corrected = "gender_preprocessed_neutral_gender.csv"

    # Shuffle the original file
    shuffle_and_save(input_file, shuffled_file)

    # Preprocess the shuffled file
    preprocess_and_save(shuffled_file, output_file_preprocessed, output_file_gender_corrected, replacement_dict)

    # Count total and unique words in each version
    total_words_original, unique_words_original = count_words(input_file)
    total_words_preprocessed, unique_words_preprocessed = count_words(output_file_preprocessed)
    total_words_gender_corrected, unique_words_gender_corrected = count_words(output_file_gender_corrected)

    print(f"Original file - Total words: {total_words_original}, Unique words: {unique_words_original}")
    print(f"Preprocessed file - Total words: {total_words_preprocessed}, Unique words: {unique_words_preprocessed}")
    print(f"Gender-corrected file - Total words: {total_words_gender_corrected}, Unique words: {unique_words_gender_corrected}")