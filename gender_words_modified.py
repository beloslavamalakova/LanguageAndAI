import pandas as pd
import re

csv_file = 'shuffled_file.csv' 
df = pd.read_csv(csv_file)

#List of gendered words and their replacements
replacement_dict = {
    r"\b(fe)?male(s)?\b": "person",
    r"\b(wo)?m[ae]n\b": "person",
    r"\b(s)?he\b": "they",
    
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
}

#Replace gendered words
def replace_gendered_words(text, replacements):
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

#Apply the replacement to the posts
df['post'] = df['post'].apply(lambda x: replace_gendered_words(str(x), replacement_dict))

#Save the modified DataFrame to a new CSV file
output_file = 'shuffled_file_gender_neutral.csv'
df.to_csv(output_file, index=False)

print(f"Modified CSV saved to {output_file}")
