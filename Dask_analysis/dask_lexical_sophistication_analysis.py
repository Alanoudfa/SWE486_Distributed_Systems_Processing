import pandas as pd
import dask.dataframe as dd
from sklearn.feature_extraction.text import TfidfVectorizer

# Objective: Analyze lexical sophistication in Arabic and English descriptions using TF-IDF scores.
# Business Impact: Understanding lexical sophistication helps businesses tailor content strategies and improve engagement by using distinctive vocabulary.

# Load the dataset
data = pd.read_csv('/Users/ruba/Desktop/processed_dataset_0_12241239_v2.csv')

# Convert the pandas DataFrame to a Dask DataFrame with one partition
df = dd.from_pandas(pd.DataFrame(data), npartitions=1)

# Initialize TF-IDF vectorizers for English and Arabic
tfidf_en = TfidfVectorizer()
tfidf_ar = TfidfVectorizer()

# Fit the TF-IDF vectorizer and transform the English and Arabic captions
tfidf_en_matrix = tfidf_en.fit_transform(df['caption_en'].compute())
tfidf_ar_matrix = tfidf_ar.fit_transform(df['caption_ar'].compute())

# Calculate the TF-IDF scores for English and Arabic captions
tfidf_en_scores = tfidf_en_matrix.sum(axis=1).A1
tfidf_ar_scores = tfidf_ar_matrix.sum(axis=1).A1

# Create Dask DataFrames for the results
result_en = dd.from_pandas(pd.DataFrame({'caption_en': df['caption_en'].compute(), 'tfidf_en': tfidf_en_scores}), npartitions=1)
result_ar = dd.from_pandas(pd.DataFrame({'caption_ar': df['caption_ar'].compute(), 'tfidf_ar': tfidf_ar_scores}), npartitions=1)

# Print the results
print(result_ar.compute())
print(result_en.compute())

# Second part: Calculate average TF-IDF scores for both languages
ddf = pd.read_csv("/Users/ruba/Desktop/processed_dataset_0_12241239_v2.csv")
ddf = dd.from_pandas(ddf, npartitions=1)

# Filter for non-null captions
ddf_filtered = ddf[ddf["caption_en"].notnull() & ddf["caption_ar"].notnull()]

# Stopwords for English and Arabic
stopwords_en = [
    "the", "and", "is", "in", "on", "with", "a", "of", "for", "to", 
    "an", "at", "it", "his", "her", "that", "there", "by", 
    "from", "its", "through", "as", "what", "this", "was", "were", 
    "be", "are", "or", "but", "if", "about", "than", "so", "we", 
    "you", "he", "she", "they", "i", "my", "me", "their", "our", 
    "your", "all", "which", "when", "where", "how", "why", "has", "had", 
    "not", "been", "can", "do", "does", "did", "some", "any", "such", "while"
]

stopwords_ar = [
    "و", "في", "على", "من", "إلى", "ب", "عن", "أن", "هذا", 
    "هذه", "ذلك", "تلك", "هو", "هي", "هم", "هن", "ما", "ماذا", 
    "لماذا", "أين", "كيف", "متى", "كل", "كان", "يكون", "كانت", 
    "ل", "لأن", "إذا", "قد", "لقد", "هل", "أو", "ثم", "أي", 
    "بعض", "عند", "منذ", "لكن", "مع", "فيها", "فيه", "بين", 
    "إلا", "حتى", "إذا", "بعد", "قبل", "أكثر", "كما", "مثل", "أيضا", 
    "الذي", "به", "التي", "ذات", "أثناء", "ذو", "أولئك", "بها"
]

# Tokenize and filter stopwords for both languages
ddf_filtered['tokens_en'] = ddf_filtered['caption_en'].str.lower().str.split()
ddf_filtered['tokens_ar'] = ddf_filtered['caption_ar'].str.lower().str.split()

ddf_filtered['filtered_en'] = ddf_filtered['tokens_en'].apply(
    lambda tokens: [word for word in tokens if word not in stopwords_en], meta=('x', 'object')
)

ddf_filtered['filtered_ar'] = ddf_filtered['tokens_ar'].apply(
    lambda tokens: [word for word in tokens if word not in stopwords_ar], meta=('x', 'object')
)

# Compute the TF-IDF matrices for filtered words
tfidf_en = TfidfVectorizer(analyzer=lambda x: x)
tfidf_en_matrix = tfidf_en.fit_transform(ddf_filtered['filtered_en'].compute()).toarray()

tfidf_ar = TfidfVectorizer(analyzer=lambda x: x)
tfidf_ar_matrix = tfidf_ar.fit_transform(ddf_filtered['filtered_ar'].compute()).toarray()

# Calculate average TF-IDF scores
avg_tfidf_en = tfidf_en_matrix.mean(axis=0).mean()
avg_tfidf_ar = tfidf_ar_matrix.mean(axis=0).mean()

print(f"Average TF-IDF score for English captions: {avg_tfidf_en}")
print(f"Average TF-IDF score for Arabic captions: {avg_tfidf_ar}")
