import pandas as pd
import re

csv_file = 'genderSeparated.csv' 
df = pd.read_csv(csv_file)

#List of gendered words and their replacements
replacement_dict = {
    r"\bwoman\b": "person",
    r"\bshe\b": "they",
    r"\bher\b": "them",
    r"\bhers\b": "their",
    r"\bherself\b": "themselves",
    r"\bgal\b": "person",
    r"\bfemale\b": "person",
    r"\bqueen\b": "person",
    r"\bgirl\b": "person",
    r"\bwife\b": "person",
    r"\bm[uo]m\b": "person",
    r"\bmother\b": "person",
    r"\blady\b": "person",
    r"\baunt\b": "person",
    r"\bniece\b": "person",
    r"\bsister\b": "person",
    r"\bdaughter\b": "person",
    r"\bdame\b": "person",
    r"\bempress\b": "person",

    r"\bman\b": "person",
    r"\bhe\b": "they",
    r"\bhim\b": "them",
    r"\bhis\b": "their",
    r"\bhimself\b": "themselves",
    r"\bking\b": "person",
    r"\bboy\b": "person",
    r"\bguy\b": "person",
    r"\bmale\b": "person",
    r"\bhusband\b": "person",
    r"\bdad\b": "person",
    r"\bfather\b": "person",
    r"\bson\b": "person",
    r"\bbrother\b": "person",
    r"\buncle\b": "person",
    r"\bnephew\b": "person",
    r"\bemperor\b": "person",
    r"\blord\b": "person",
}

#Replace gendered words
def replace_gendered_words(text, replacements):
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

#Apply the replacement to the posts
df['post'] = df['post'].apply(lambda x: replace_gendered_words(str(x), replacement_dict))

#Save the modified DataFrame to a new CSV file
output_file = 'gender_words_modified.csv'
df.to_csv(output_file, index=False)

print(f"Modified CSV saved to {output_file}")
