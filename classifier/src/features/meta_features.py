""" Extract meta features from the data """

import pandas as pd
from transformers import pipeline, AutoTokenizer
import spacy
from tqdm import tqdm
from typing import Optional
import re
import csv
tqdm.pandas()


def has_title(data: pd.DataFrame) -> pd.DataFrame:
    """
    Check if the data has a title.
    args:
        data: pd.DataFrame
        the title is a string
    returns:
        a boolean value, True if the title is not null, False otherwise
    """
    return data["title"].notna()


def has_description(data: pd.DataFrame) -> pd.DataFrame:
    """
    Check if the data has a description.
    args:
        data: pd.DataFrame
        the description is a string
    returns:
        a boolean value, True if the description is not null, False otherwise
    """
    return data["description"].notna()


def video_duration(data: pd.DataFrame) -> pd.DataFrame:
    """
    Check the duration of the video.
    args:
        data: pd.DataFrame
        the duration format is "PT3M8S" for 3 minutes and 8 seconds 
        or "PT1H23M45S" for 1 hour, 23 minutes, and 45 seconds
    returns:
        the duration of the video in seconds
    """
    return data["duration"].apply(lambda x: pd.to_timedelta(x).total_seconds())


def description_length(data: pd.DataFrame) -> pd.DataFrame:
    """
    Check the length of the description.
    
    calculate the number of words in the description, exclude the punctuation and special characters
    args:
        data: pd.DataFrame
        the description is a string
    returns:
        the number of words in the description
    """
    return data["description"].str.replace(r'[^\w\s]', '', regex=True).str.strip().apply(lambda x: len(x.split()) if pd.notna(x) and x else 0)


def description_unique_words(data: pd.DataFrame) -> pd.DataFrame:
    """
    Check the unique words in the description.
    
    args:
        data: pd.DataFrame
        the description is a string
    returns:
        the number of unique words in the description
    """
    return data["description"].str.replace(r'[^\w\s]', '', regex=True).str.strip().apply(lambda x: len(set(x.split())) if pd.notna(x) and x else 0)


def view_count(data: pd.DataFrame) -> pd.DataFrame:
    """
    Check the view count of the video.
    args:
        data: pd.DataFrame
        the view count is a integer
    returns:
        the view count of the video
    """
    return data["view_count"]


def like_count(data: pd.DataFrame) -> pd.DataFrame:
    """
    Check the like count of the video.
    args:
        data: pd.DataFrame
        the like count is a integer
    returns:
        the like count of the video
    """
    return data["like_count"]


def comment_count(data: pd.DataFrame) -> pd.DataFrame:
    """
    Check the comment count of the video.
    args:
        data: pd.DataFrame
        the comment count is a integer
    returns:
        the comment count of the video
    """
    return data["comment_count"]


def subscriber_count(data: pd.DataFrame) -> pd.DataFrame:
    """
    Check the subscriber count of the video.
    args:
        data: pd.DataFrame
        the subscriber count is a integer
    returns:
        the subscriber count of the video
    """
    return data["subscriber_count"]


def channel_upload_count(data: pd.DataFrame) -> pd.DataFrame:
    """
    Check the upload count of the channel.
    args:
        data: pd.DataFrame
        the upload count is a integer
    returns:
        the upload count of the channel
    """
    return data["video_count"]
    

class MetaFeatureExtractorNLP:
    
    """
    Extract meta features from the data using LLM.
    """
    def __init__(self,
                 sentiment_model: str = "tabularisai/multilingual-sentiment-analysis"):
        
        self.sentiment_model = sentiment_model
        self.sentiment_pipeline = pipeline("sentiment-analysis", 
                                           model=sentiment_model)
        self.tokenizer = AutoTokenizer.from_pretrained(sentiment_model)
        self.nlp = spacy.load("en_core_web_sm")
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing URLs and special characters.
        args:
            text: str - input text
        returns:
            str - cleaned text
        """
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove URLs (http, https, www)
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, '', text)
        
        # Remove email addresses
        email_pattern = r'\S+@\S+'
        text = re.sub(email_pattern, '', text)
        
        # Remove special characters but keep letters, numbers, spaces, and basic punctuation
        # Keep: letters, numbers, spaces, periods, commas, exclamation, question marks
        text = re.sub(r'[^\w\s.,!?]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
        
    def extract_sentiment(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the sentiment of the data.
        args:
            data: pd.DataFrame
        returns:

            the sentiment of the data
            Number of Classes: 5 (Very Negative, Negative, Neutral, Positive, Very Positive)
            {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
        """
        print("📊 Extracting sentiment features...")
        def process_sentiment(text):
            if text is None or pd.isna(text):
                return None
            try:
                # Clean text: remove URLs and special characters
                cleaned_text = self.clean_text(text)
                
                if not cleaned_text:
                    return None
                
                # Truncate text using tokenizer to ensure it fits model's max length
                max_length = 512
                # Tokenize and truncate - get token IDs
                encoded = self.tokenizer.encode(cleaned_text, max_length=max_length, 
                                               truncation=True, return_tensors=None)
                # Decode back to get properly truncated text
                truncated_text = self.tokenizer.decode(encoded, skip_special_tokens=True)
                
                # Get sentiment result (pipeline returns a list with dict)
                result = self.sentiment_pipeline(truncated_text)
                if result and isinstance(result, list) and len(result) > 0:
                    return result[0].get("label", None)
                return None
            except Exception as e:
                # Fallback: simple character truncation if tokenizer fails
                try:
                    cleaned_text = self.clean_text(text)
                    if not cleaned_text:
                        return None
                    if len(cleaned_text) > 500:
                        cleaned_text = cleaned_text[:500]
                    result = self.sentiment_pipeline(cleaned_text)
                    if result and isinstance(result, list) and len(result) > 0:
                        return result[0].get("label", None)
                    return None
                except:
                    return None
        
        return data["description"].progress_apply(process_sentiment)
        
    def sentence_count(self, 
                        data: pd.DataFrame,
                        doc: Optional[object] = None) -> pd.DataFrame:
        """
        Extract the sentence length of the data.
        args:
            data: pd.DataFrame
            doc: spacy.Doc
        returns:
            the sentence length of the data
        """
        
        try:
            if doc is None:
                print("📝 Processing sentences with spaCy...")
                # Deal with the case where the description is None or NaN
                def process_nlp(text):
                    if text is None or pd.isna(text):
                        return None
                    try:
                        return self.nlp(str(text))
                    except:
                        return None
                doc_df = data["description"].progress_apply(process_nlp)
            print("📊 Counting sentences...")
            sentences_df = doc_df.progress_apply(lambda x: list(x.sents) if x is not None else None)
            return sentences_df.apply(lambda x: len(x) if x is not None else 0)

        except Exception as e:
            print(f"Error extracting sentence length: {e}")
            return pd.Series([0] * len(data), index=data.index)
    

def extract_meta_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the meta features from the data.
    args:
        data: pd.DataFrame
    returns:
        the meta features of the data
    """
    print(f"🚀 Starting meta feature extraction for {len(data)} videos...")
    data.set_index("video_id", inplace=True)
    data.sort_index(inplace=True)
    
    print("📋 Extracting basic features...")
    has_title_df = has_title(data)
    has_description_df = has_description(data)
    video_duration_df = video_duration(data)
    description_length_df = description_length(data)
    description_unique_words_df = description_unique_words(data)
    view_count_df = view_count(data)
    like_count_df = like_count(data)
    comment_count_df = comment_count(data)
    subscriber_count_df = subscriber_count(data)
    channel_upload_count_df = channel_upload_count(data)
    
    print("🤖 Initializing NLP models...")
    nlp_obj = MetaFeatureExtractorNLP()
    sentiment_df = nlp_obj.extract_sentiment(data)
    sentence_count_df = nlp_obj.sentence_count(data)

    print("🔗 Combining all features...")
    # Ensure all Series have names for proper concatenation
    feature_dict = {
        'description': data["description"],
        'has_title': has_title_df,
        'has_description': has_description_df,
        'video_duration': video_duration_df,
        'description_length': description_length_df,
        'description_unique_words': description_unique_words_df,
        'view_count': view_count_df,
        'like_count': like_count_df,
        'comment_count': comment_count_df,
        'subscriber_count': subscriber_count_df,
        'channel_upload_count': channel_upload_count_df,
        'sentiment': sentiment_df,
        'sentence_count': sentence_count_df
    }
    
    # Convert all to Series with names and ensure same index
    features_list = []
    for name, series in feature_dict.items():
        if not isinstance(series, pd.Series):
            series = pd.Series(series, index=data.index, name=name)
        else:
            series.name = name
        features_list.append(series)
    
    result = pd.concat(features_list, axis=1)
    # Ensure index has the name 'video_id' so it's saved with the correct column name
    result.index.name = 'video_id'
    print("✅ Feature extraction completed!")
    return result
    

if __name__ == "__main__":
    print("📂 Loading data...")
    data = pd.read_excel("data/youtube_database.xlsx", sheet_name="CredibleVideos")
    print(f"✅ Loaded {len(data)} records")
    
    meta_features = extract_meta_features(data)
    
    print("💾 Saving results to CSV...")
    # Clean description field: replace newlines with spaces for better CSV readability
    if 'description' in meta_features.columns:
        meta_features['description'] = meta_features['description'].astype(str).str.replace('\n', ' ', regex=False).str.replace('\r', ' ', regex=False)
    
    # Use index=True to save video_id (which is the index) as a column
    # Use quoting=csv.QUOTE_NONNUMERIC to ensure all string fields are quoted
    meta_features.to_csv("data/meta_features.csv", index=True, 
                        quoting=csv.QUOTE_NONNUMERIC, escapechar='\\')
    print("✅ Meta features extracted and saved successfully!")
    