"""
Business Question: Keyword Frequency Count (4.1)
Objective: Identify the most frequent keywords in both Arabic and English descriptions.

Business Impact: 
This business goal helps businesses tailor messaging to regional audiences. 
Understanding which terms are most common allows businesses to enhance their content’s visibility 
in searches and improve localization by aligning descriptions with cultural relevance.
"""


from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Keyword Frequency Count") \
    .getOrCreate()

# Load the dataset 
dataset_path = "/path/to/your/dataset.csv"  # Replace with your actual dataset path
df = spark.read.csv(dataset_path, header=True, inferSchema=True)

# Check if the dataset is loaded correctly by showing the first 5 rows
df.show(5)

# Tokenize the English descriptions
tokenized_en = df.select(explode(split(df["caption_en"], " ")).alias("word_en")).cache()

# Tokenize the Arabic descriptions
tokenized_ar = df.select(explode(split(df["caption_ar"], " ")).alias("word_ar")).cache()

# Count the frequency of English words
word_count_en = tokenized_en.groupBy("word_en").count().orderBy("count", ascending=False)

# Count the frequency of Arabic words
word_count_ar = tokenized_ar.groupBy("word_ar").count().orderBy("count", ascending=False)

# English stopwords list
stopwords_en = [
    "the", "and", "is", "in", "on", "with", "a", "of", "for", "to", 
    "an", "at", "it", "t", "his", "her", "that", "there", "by", 
    "from", "its", "through", "as", "what", "this", "was", "were", 
    "be", "are", "or", "but", "if", "about", "than", "so", "we", 
    "you", "he", "she", "they", "i", "my", "me", "their", "our", 
    "your", "all", "which", "when", "where", "how", "why", "has", "had", 
    "not", "been", "can", "do", "does", "did", "some", "any", "such", "while"
]

# Filter out English stop words
filtered_word_count_en = word_count_en.filter(~word_count_en.word_en.isin(stopwords_en))

# Arabic stopwords list
stopwords_ar = [
    "و", "في", "على", "من", "إلى", "ب", "عن", "أن", "هذا", 
    "هذه", "ذلك", "تلك", "هو", "هي", "هم", "هن", "ما", "ماذا", 
    "لماذا", "أين", "كيف", "متى", "كل", "كان", "يكون", "كانت", 
    "ل", "لأن", "إذا", "قد", "لقد", "هل", "أو", "ثم", "أي", 
    "بعض", "عند", "منذ", "لكن", "مع", "فيها", "فيه", "بين", 
    "إلا", "حتى", "إذا", "بعد", "قبل", "أكثر", "كما", "مثل", "أيضا", 
    "الذي", "به", "التي", "ذات", "أثناء", "ذو", "أولئك", "بها"
]

# Filter out Arabic stop words
filtered_word_count_ar = word_count_ar.filter(~word_count_ar.word_ar.isin(stopwords_ar))

# Show the top 10 most frequent English and Arabic words
filtered_word_count_en.show(10)
filtered_word_count_ar.show(10)

# Save results to CSV (optional)
filtered_word_count_en.write.csv("keyword_frequency_en.csv", header=True)
filtered_word_count_ar.write.csv("keyword_frequency_ar.csv", header=True)

# Stop Spark session
spark.stop()
