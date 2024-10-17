"""
Business Question: 4.2 Compare the Complexity of Descriptions - Sentence Length
Objective: Compare the complexity of Arabic and English text descriptions by analyzing sentence length.

Business Impact:
This analysis helps businesses understand the different content styles preferred by regional audiences. By adapting sentence length to suit various customer segments, businesses can improve readability and user engagement, leading to more effective localized content.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, size, split

# Initialize Spark session
spark = SparkSession.builder.appName("Avg-Sentence-Word-Count").getOrCreate()

# Load the dataset (modify the path to your dataset as necessary)
df = spark.read.csv("/path/to/your/dataset.csv", header=True, inferSchema=True)

# Filter out null values in both languages
df_filtered = df.filter(df["caption_en"].isNotNull() & df["caption_ar"].isNotNull())

# Add columns for word count in English and Arabic captions
df_lengths = df_filtered.withColumn("en_word_count", size(split(col("caption_en"), " "))) \
                        .withColumn("ar_word_count", size(split(col("caption_ar"), " ")))

# Calculate the average word count for both languages
avg_lengths = df_lengths.select(
    avg("en_word_count").alias("English Avg Sentence Length"),
    avg("ar_word_count").alias("Arabic Avg Sentence Length")
)

# Show the result
avg_lengths.show()

# Stop the Spark session
spark.stop()