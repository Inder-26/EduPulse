
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import os

def verify_eda():
    print("Verifying Extended EDA step...")
    
    # 1. Load Data
    data_path = '../data/all_reviews_merged.csv'
    if not os.path.exists(data_path):
        # Fallback for running from root
        data_path = 'data/all_reviews_merged.csv'
        
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    try:
        df = pd.read_csv(data_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Missing Values
    df['content'] = df['content'].fillna('')
    print("Missing values handled.")

    # 3. Feature Engineering
    # Text-based
    df['content'] = df['content'].astype(str)
    df['review_length'] = df['content'].apply(len)
    df['word_count'] = df['content'].apply(lambda x: len(x.split()))
    
    def get_sentiment(text):
        return TextBlob(text).sentiment.polarity

    # Test sentiment on a small subset for speed
    df['sentiment_polarity'] = df['content'].head(10).apply(get_sentiment)
    print("Text features (length, word_count, sentiment) created.")

    # Time-based
    if 'created' in df.columns:
        df['created'] = pd.to_datetime(df['created'], errors='coerce', utc=True)
        df['year'] = df['created'].dt.year
        print("Time features (year) created.")
    
    # 4. Check for WordCloud import
    wc = WordCloud()
    print("WordCloud import successful.")

    print("Verification completed successfully.")

if __name__ == "__main__":
    verify_eda()
