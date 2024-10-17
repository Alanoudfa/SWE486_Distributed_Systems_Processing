
# Objective: Analyze lexical sophistication in Arabic and English descriptions using TF-IDF scores.
# Business Impact: Understanding lexical sophistication helps businesses tailor content strategies and improve engagement by using distinctive vocabulary.

import dask.dataframe as dd
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# Load the Dataset 
df = spark.read.csv("/path/to/dataset.csv", header=True, inferSchema=True)

# Drop rows with missing values in Arabic and English captions
df = df.dropna(subset=['caption_ar', 'caption_en'])

# Get the first 10 Arabic and English captions for analysis
captions_ar = df['caption_ar'].head(10)  # No need to use compute() after head()
captions_en = df['caption_en'].head(10)  # No need to use compute() after head()

# Initialize the TF-IDF vectorizer for both languages
tfidf_ar = TfidfVectorizer()
tfidf_en = TfidfVectorizer()

# Fit and transform the first 10 Arabic and English captions using TF-IDF
tfidf_ar_matrix = tfidf_ar.fit_transform(captions_ar)
tfidf_en_matrix = tfidf_en.fit_transform(captions_en)

# Convert the TF-IDF matrices to arrays for easier manipulation
tfidf_ar_array = tfidf_ar_matrix.toarray()
tfidf_en_array = tfidf_en_matrix.toarray()

# Create a DataFrame for Arabic captions and their corresponding TF-IDF scores
result_ar_df = pd.DataFrame({
    'caption_ar': captions_ar,
    'tfidf_ar': list(tfidf_ar_array)  # Store TF-IDF scores in a list
})

# Create a DataFrame for English captions and their corresponding TF-IDF scores
result_en_df = pd.DataFrame({
    'caption_en': captions_en,
    'tfidf_en': list(tfidf_en_array)  # Store TF-IDF scores in a list
})

# Function to print TF-IDF results in a structured format
def print_tfidf_results(df, caption_col, tfidf_col):
    # Print the header
    print("+{:-<40}+{:-<70}+".format('', ''))
    print("| {:<38} | {:<68} |".format(caption_col, tfidf_col))
    print("+{:-<40}+{:-<70}+".format('', ''))
    
    # Iterate over the DataFrame and print each caption with its TF-IDF scores
    for i, row in df.iterrows():
        caption = row[caption_col]
        tfidf_values = np.round(row[tfidf_col], 6)  # Round TF-IDF values for neatness
        tfidf_str = ', '.join(map(str, tfidf_values.tolist()))  # Convert TF-IDF values to string
        print("| {:<38} | {:<68} |".format(caption, tfidf_str))
        print("+{:-<40}+{:-<70}+".format('', ''))

# Print TF-IDF results for Arabic captions
print("TF-IDF Results for Arabic Captions:")
print_tfidf_results(result_ar_df, 'caption_ar', 'tfidf_ar')

# Print TF-IDF results for English captions
print("\nTF-IDF Results for English Captions:")
print_tfidf_results(result_en_df, 'caption_en', 'tfidf_en')
