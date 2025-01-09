"""
Script: Gender-Neutral Text Preprocessor
Author: Dani
Date: 7/1/25

Description:
This script processes a CSV file containing text posts to replace gendered words 
with gender-neutral alternatives. It reads the input CSV file, applies regex-based 
replacements to the 'post' column, and outputs a new CSV file with the modified text.

Key Features:
- Supports a comprehensive list of gendered words and their neutral replacements.
- Uses regex for flexible and efficient text pattern matching.
- Processes large CSV files containing text data.

Inputs and Outputs:
- Input: A CSV file containing a 'post' column with text data to be processed.
- Output: A new CSV file where all gendered words in the 'post' column have been replaced with gender-neutral terms.

Notes:
- Ensure the input CSV file has a 'post' column that contains the text data.
- Works with altered gender CSV file where all of the information is in 1 column separated with commas (done that manually in excel)
- The replacement dictionary can be expanded to include additional terms as needed.
"""

import pandas as pd
import re

# Specify the input CSV file path
csv_file = 'shuffled_file_preprocessed.csv' 
df = pd.read_csv(csv_file)

# Dictionary of gendered words and their replacements
replacement_dict = {
    # Patterns for terms for both genders
    r"\b(fe)?male(s)?\b": "person",
    r"\b(wo)?m[ae]n\b": "person",
    r"\b(s)?he\b": "they",
    r"\bpolice(wo)?m[ae]n\b": "person",
    r"\bpost(wo)?m[ae]n\b": "person",
    
    # Patterns for female-specific terms
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

    # Patterns for male-specific terms
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

# Replace gendered words
def replace_gendered_words(text, replacements):
    """
    Replaces gendered words in the given text with their neutral equivalents 
    based on the replacement dictionary.
    
    Args:
        text (str): The input text to process.
        replacements (dict): Dictionary of patterns and their replacements.
    
    Returns:
        str: The processed text with gendered words replaced.
    """
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

#Apply the replacement to the posts
df['post'] = df['post'].apply(lambda x: replace_gendered_words(str(x), replacement_dict))

#Save the modified DataFrame to a new CSV file
output_file = 'shuffled_file_preprocessed_gender_neutral.csv'
df.to_csv(output_file, index=False)

print(f"Modified CSV saved to {output_file}")
