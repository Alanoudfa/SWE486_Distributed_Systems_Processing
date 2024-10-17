import dask.dataframe as dd
from dask.distributed import Client

# Objective: Calculate the average sentence length for Arabic and English descriptions.
# Business Impact: Understanding sentence length helps assess the complexity of content, informing localization strategies.

if __name__ == '__main__':
    # Initialize a Dask client for distributed computing
    client = Client()

    # Load the dataset into a Dask DataFrame
    df = dd.read_csv("/Users/ruba/Desktop/processed_dataset_0_12241239_v2.csv")

    # Filter out rows where either English or Arabic captions are missing
    df_filtered = df.dropna(subset=['caption_en', 'caption_ar'])

    # Calculate word counts for both English and Arabic captions
    df_lengths = df_filtered.assign(
        en_word_count=df_filtered['caption_en'].str.split().str.len(),  # Count words in English captions
        ar_word_count=df_filtered['caption_ar'].str.split().str.len()   # Count words in Arabic captions
    )

    # Compute the average word counts for both languages
    avg_lengths = df_lengths[['en_word_count', 'ar_word_count']].mean().compute()

    # Print the average sentence lengths for both languages
    print("English Avg Sentence Length:", avg_lengths['en_word_count'])
    print("Arabic Avg Sentence Length:", avg_lengths['ar_word_count'])

    # Close the Dask client
    client.close()
