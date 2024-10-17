"""
Business Question: 4.2 Compare the Complexity of Descriptions - Lexical Sophistication
Objective: Compare the complexity of Arabic and English text descriptions using the lexical sophistication factor, which refers to the use of advanced or less common vocabulary.

Business Impact:
This analysis helps businesses understand the linguistic complexity required to engage their audience. By tailoring vocabulary sophistication to the target market, businesses can enhance their content’s appeal and alignment with audience preferences, improving localization strategies.
"""
# Import necessary modules
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.sql.functions import col, expr
from pyspark.ml.functions import vector_to_array

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("TF-IDF Lexical Sophistication Analysis") \
    .getOrCreate()

# Load the Dataset (Modify the path as necessary)
df = spark.read.csv("/path/to/dataset.csv", header=True, inferSchema=True)

# Filter out rows with null captions
df_filtered = df.filter(df["caption_en"].isNotNull() & df["caption_ar"].isNotNull())

# Tokenize English and Arabic captions
tokenizer_en = Tokenizer(inputCol="caption_en", outputCol="tokens_en")
tokenizer_ar = Tokenizer(inputCol="caption_ar", outputCol="tokens_ar")

df_en_tokenized = tokenizer_en.transform(df_filtered)
df_ar_tokenized = tokenizer_ar.transform(df_filtered)

# Define stopwords for English and Arabic
stopwords_en = [
    "the", "and", "is", "in", "on", "with", "a", "of", "for", "to", 
    "an", "at", "it", "t", "his", "her", "that", "there", "by", 
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

# Remove stopwords for English and Arabic
remover_en = StopWordsRemover(inputCol="tokens_en", outputCol="filtered_en", stopWords=stopwords_en, locale="en_US")
remover_ar = StopWordsRemover(inputCol="tokens_ar", outputCol="filtered_ar", stopWords=stopwords_ar, locale="ar")

df_en_filtered = remover_en.transform(df_en_tokenized)
df_ar_filtered = remover_ar.transform(df_ar_tokenized)

# Apply CountVectorizer (Term Frequency)
vectorizer_en = CountVectorizer(inputCol="filtered_en", outputCol="tf_en", vocabSize=1000)
vectorizer_ar = CountVectorizer(inputCol="filtered_ar", outputCol="tf_ar", vocabSize=1000)

cv_model_en = vectorizer_en.fit(df_en_filtered)
cv_model_ar = vectorizer_ar.fit(df_ar_filtered)

df_en_tf = cv_model_en.transform(df_en_filtered)
df_ar_tf = cv_model_ar.transform(df_ar_filtered)

# Apply IDF (Inverse Document Frequency)
idf_en = IDF(inputCol="tf_en", outputCol="tfidf_en")
idf_ar = IDF(inputCol="tf_ar", outputCol="tfidf_ar")

idf_model_en = idf_en.fit(df_en_tf)
idf_model_ar = idf_ar.fit(df_ar_tf)

df_en_tfidf = idf_model_en.transform(df_en_tf)
df_ar_tfidf = idf_model_ar.transform(df_ar_tf)

# Convert the TF-IDF vector into an array
df_en_tfidf_array = df_en_tfidf.withColumn("tfidf_en_array", vector_to_array(col("tfidf_en")))
df_ar_tfidf_array = df_ar_tfidf.withColumn("tfidf_ar_array", vector_to_array(col("tfidf_ar")))

# Calculate the average TF-IDF by flattening the array and averaging its elements
df_en_avg = df_en_tfidf_array.selectExpr("explode(tfidf_en_array) as tfidf_value") \
    .selectExpr("avg(tfidf_value) as avg_tfidf_en")
df_ar_avg = df_ar_tfidf_array.selectExpr("explode(tfidf_ar_array) as tfidf_value") \
    .selectExpr("avg(tfidf_value) as avg_tfidf_ar")

# Show the overall average TF-IDF for English and Arabic
df_en_avg.show()
df_ar_avg.show()

# Stop the Spark session
spark.stop()