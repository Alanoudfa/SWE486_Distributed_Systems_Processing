# Objective: Identify the most frequent keywords in both Arabic and English descriptions.
# Business Impact: Understanding keyword frequency helps businesses enhance content visibility and improve localization strategies by identifying relevant terms.

import dask.bag as db
import pandas as pd
import re



def preprocess(text):
    # Remove punctuation and digits, strip whitespace, and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip().lower()
    return text

def filter_stopwords(word_list, stop_words):
    # Filter out stop words and words with less than 2 characters
    return [word for word in word_list if word not in stop_words and len(word) > 1]

def main():
    # Define Arabic stop words
    arabic_stopwords = [
        "و", "في", "على", "من", "إلى", "ب", "عن", "أن", "هذا",
        "هذه", "ذلك", "تلك", "هو", "هي", "هم", "هن", "ما", "ماذا",
        "لماذا", "أين", "كيف", "متى", "كل", "كان", "يكون", "كانت",
        "ل", "لأن", "إذا", "قد", "لقد", "هل", "أو", "ثم", "أي",
        "بعض", "عند", "منذ", "لكن", "مع", "فيها", "فيه", "بين",
        "إلا", "حتى", "إذا", "بعد", "قبل", "أكثر", "كما", "مثل", "أيضا",
        "الذي", "به", "التي", "ذات", "أثناء", "ذو", "أولئك", "بها", "فوق"
    ]

    # Define English stop words
    english_stopwords = set([
        "the", "is", "in", "it", "of", "and", "to", "a", "that", "i", "you",
        "he", "she", "we", "they", "on", "for", "this", "with", "as", "at",
        "or", "but", "by", "an", "be", "was", "were", "so", "if", "no", "yes"
    ])

    # Load the dataset as a Dask Bag
    text = db.read_text('/Users/ruba/Desktop/processed_dataset_0_12241239_v2.csv')

    # Process the text: preprocess, split into words, and get word frequencies
    words = (text
             .map(preprocess)                   # Preprocess each line of text
             .map(lambda x: x.split())         # Split the text into words
             .flatten()                         # Flatten the list of lists into a single list
             .frequencies(sort=True))           # Count word frequencies

    # Filter words to separate English and Arabic
    english_words = words.filter(lambda x: re.match(r'[a-zA-Z]+', x[0]))  # Match English words
    arabic_words = words.filter(lambda x: re.match(r'[ء-ي]+', x[0]))       # Match Arabic words

    # Filter out stop words from the frequency lists
    english_filtered = english_words.filter(lambda x: x[0] not in english_stopwords).take(10)
    arabic_filtered = arabic_words.filter(lambda x: x[0] not in arabic_stopwords).take(10)

    # Convert the filtered results to DataFrames for better presentation
    english_df = pd.DataFrame(english_filtered, columns=["Word", "Frequency"])
    arabic_df = pd.DataFrame(arabic_filtered, columns=["Word", "Frequency"])

    # Print the top 10 most frequent words for both languages
    print("Top 10 English words:")
    print(english_df)

    print("Top 10 Arabic words:")
    print(arabic_df)

if __name__ == "__main__":
    main()
