# youtube_data_pipeline.py

import os
import sqlite3
import re
import time

import pandas as pd
from typing import Tuple

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter

from credibility import flag_credible_channels
from credible_videos import create_credible_videos_table
from captioning import fetch_and_save_captions
from scoring import score_videos

DB_NAME = 'youtube_data.db'

# NOTE: Downloading captions requires OAuth authentication (SCOPES and CLIENT_SECRETS_FILE).
# The captions().download() API is only available to authenticated users.
# If you don't need to download the full text, you can revert to using just the API_KEY.
# If you use the authenticated path, the API_KEY is not strictly needed for the build function.

API_KEY = "AIzaSyAK-PTFjHh6qz_Xwl7shmqhafjAtUp_JwM" #os.environ['YOUTUBE_API_KEY']
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'
CLIENT_SECRETS_FILE = None
MAX_RESULTS_PER_SEARCH = 30
CREDIBILITY_MASTER_FILE = r"Data\Transplants_Team_Master_File.xlsx"

rate_limiter = InMemoryRateLimiter(
    requests_per_second=66,
    check_every_n_seconds=0.1,
    max_bucket_size=5,
)

gemini_api_key = API_KEY#os.environ['GEMINI_API_KEY']
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=gemini_api_key, rate_limiter=rate_limiter)

print(f"Using this GEMINI API KEY: {gemini_api_key}")

keys_1 = set(pd.read_csv(r'Keywords\relevant_keywords.csv')['search key'])
keys_2 = set(pd.read_csv(r'Keywords\reddit_posts_keywords_dedup.csv')['keywords'])
union_set = keys_1.union(keys_2)

terms = list(union_set)

# --- AUTHENTICATION ---

def authenticate_youtube_service():
    """Authenticates using OAuth 2.0 flow to get a service object."""
    try:
        # Note: This will require you to go through the browser-based authentication flow.
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_console()
        return build(API_SERVICE_NAME, API_VERSION, credentials=credentials)
    except FileNotFoundError:
        print(f"WARNING: Client secrets file '{CLIENT_SECRETS_FILE}' not found. Falling back to Developer Key.")
        print("Caption downloading will fail without proper OAuth authentication.")
        return build(API_SERVICE_NAME, API_VERSION, developerKey=API_KEY)
    except Exception as e:
        print(f"Authentication failed: {e}. Falling back to Developer Key.")
        return build(API_SERVICE_NAME, API_VERSION, developerKey=API_KEY)

# --- DATABASE UTILS ---

def create_db_tables(conn):
    """
    Creates the Channels, Videos, and SearchTerms tables.
    """
    cursor = conn.cursor()
    
    # SearchTerms Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS SearchTerms (
        term TEXT PRIMARY KEY NOT NULL,
        search_started INTEGER DEFAULT 0
    );
    """)
    
    # Channels Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Channels (
        channel_id TEXT PRIMARY KEY NOT NULL,
        handle TEXT,
        title TEXT,
        published_at TEXT,
        subscriber_count INTEGER,
        video_count INTEGER,
        view_count INTEGER,
        is_credible INTEGER DEFAULT 0
    );
    """)
    
    # Videos Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Videos (
        video_id TEXT PRIMARY KEY NOT NULL,
        channel_id TEXT NOT NULL,
        search_term TEXT,
        title TEXT,
        description TEXT,
        published_at TEXT,
        duration TEXT,
        view_count INTEGER,
        like_count INTEGER,
        comment_count INTEGER,
        FOREIGN KEY (channel_id) REFERENCES Channels (channel_id)
    );
    """)
    
    conn.commit()
    print(f"Database tables created or verified in '{DB_NAME}'.")

# --- YOUTUBE API UTILS ---

def get_channel_data(youtube, channel_id):
    """Fetches key statistics and snippet for a given channel ID."""
    try:
        channels_list_response = youtube.channels().list(
            part='snippet,statistics',
            id=channel_id
        ).execute()
        if channels_list_response.get("items"):
            return channels_list_response["items"][0]
    except HttpError as e:
        print(f"Could not retrieve channel data for {channel_id}: {e}")
    return {}

def get_video_details(youtube, video_id):
    """Fetches detailed information for a given video ID."""
    try:
        results = youtube.videos().list(
            # Requesting only the parts needed for the schema
            part='contentDetails,snippet,statistics',
            id=video_id
        ).execute()
        
        if results.get("items"):
            return results["items"][0]
    except HttpError as e:
        print(f"Could not retrieve video details for {video_id}: {e}")
    return {}

def list_captions(youtube, video_id):
    """Fetches caption track metadata for a given video ID."""
    try:
        results = youtube.captions().list(
            part="snippet",
            videoId=video_id
        ).execute()
        return results.get("items", [])
    except HttpError as e:
        # A common error here is the API not being enabled for the project
        print(f"Could not retrieve caption metadata for video {video_id}: {e}")
        return []

def download_caption_text(youtube, caption_id):
    """
    Downloads the full caption text using the authenticated service.
    NOTE: Requires OAuth 2.0 authentication.
    """
    try:
        # Use 'srt' (SubRip) format for easy parsing of the text content.
        request = youtube.captions().download(id=caption_id, tfmt='srt', prettyPrint=False)
        response = request.execute()
        
        # The response is usually a raw string of the subtitle file content.
        # We clean it up to store only the text (removing timestamps/indexes).
        text = response.replace('\r', '').replace('\n', ' ')
        
        # Simple regex to remove subtitle index, timestamps, and empty lines
        # This is a basic cleanup and may need refinement for perfect results.
        cleaned_text = re.sub(r'(\d+\s+)(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\s+)', '', text)
        
        return ' '.join(cleaned_text.split()) # Normalize whitespace
        
    except HttpError as e:
        print(f"WARNING: Failed to download caption {caption_id} (Need OAuth 2.0 and correct API setup): {e}")
        return None

def try_fetch_transcript(video_id: str, languages: Tuple[str, ...] = ("en", "en-US", "en-GB")) -> Tuple[bool, str]:
    """
    Works with both legacy (<1.x) and new (>=1.x) youtube-transcript-api.
    Returns (ok, text_or_error).
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
    except Exception:
        return (False, "")

    try:
        langs = list(languages)
        # Legacy API provided a static method; newer API provides an instance with .fetch()
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            data = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
        else:
            api = YouTubeTranscriptApi()
            fetched = api.fetch(video_id, languages=langs)
            if hasattr(fetched, "to_dict"):
                data = fetched.to_dict()
            elif hasattr(fetched, "to_raw_data"):
                data = fetched.to_raw_data()
            else:
                data = list(fetched)

        text = " ".join(d.get("text", "") for d in data if isinstance(d, dict) and d.get("text")).strip()
        return (True, text if text else "")
    except (NoTranscriptFound, TranscriptsDisabled):
        return (False, "")
    except Exception as e:
        return (False, f"")

# --- DATABASE INSERTION LOGIC ---

# Function to populate the SearchTerms table
def populate_search_terms(conn, terms_list):
    """Inserts all terms from the initial list into the SearchTerms table."""
    cursor = conn.cursor()

    records_to_insert = [(term,) for term in terms_list]

    cursor.executemany("INSERT OR IGNORE INTO SearchTerms (term) VALUES (?)", records_to_insert)
    conn.commit()
    print(f"Populated/Verified {len(records_to_insert)} terms in the SearchTerms table.")

def insert_channel(conn, channel_data):
    """Inserts a channel, ignoring duplicates (production-style)."""
    if not channel_data: return
    
    channel_id = channel_data['id']
    snippet = channel_data['snippet']
    stats = channel_data['statistics']
    
    channel_record = (
        channel_id,
        snippet.get('customUrl'),
        snippet.get('title'),
        snippet.get('publishedAt'),
        int(stats.get('subscriberCount', 0)),
        int(stats.get('videoCount', 0)),
        int(stats.get('viewCount', 0)),
    )
    
    # INSERT OR IGNORE: Prevents errors on duplicate channel_id (PRIMARY KEY)
    sql = """
    INSERT OR IGNORE INTO Channels 
    (channel_id, handle, title, published_at, subscriber_count, video_count, view_count) 
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    conn.execute(sql, channel_record)
    conn.commit()

def insert_video(conn, video_data, search_term):
    """Inserts a video, ignoring duplicates."""
    if not video_data: return
    
    video_id = video_data['id']
    snippet = video_data['snippet']
    stats = video_data['statistics']
    content_details = video_data['contentDetails']

    video_record = (
        video_id,
        snippet.get('channelId'),
        search_term,
        snippet.get('title'),
        snippet.get('description'),
        snippet.get('publishedAt'),
        content_details.get('duration'),
        int(stats.get('viewCount', 0)),
        int(stats.get('likeCount', 0)),
        int(stats.get('commentCount', 0))
    )

    sql = """
    INSERT OR IGNORE INTO Videos 
    (video_id, channel_id, search_term, title, description, published_at, duration, view_count, like_count, comment_count) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    conn.execute(sql, video_record)
    conn.commit()

def insert_caption(conn, caption_meta, caption_text, video_id):
    """Inserts a caption record, including the full text."""
    if not caption_meta: return
    
    caption_id = caption_meta['id']
    snippet = caption_meta['snippet']
    
    caption_record = (
        caption_id,
        video_id,
        snippet.get('language'),
        1 if snippet.get('trackKind') == 'ASR' else 0, # Auto-generated (ASR)
        caption_text,
    )
    
    sql = """
    INSERT OR IGNORE INTO Captions 
    (caption_id, video_id, language, is_auto_generated, caption_text) 
    VALUES (?, ?, ?, ?, ?)
    """
    conn.execute(sql, caption_record)
    conn.commit()

# --- MAIN EXECUTION FLOW ---

def update_youtube_database(youtube, conn):
    """
    The main production logic loop: searches, fetches details, and updates the database.
    """
    cursor = conn.cursor()

    # *** Fetch terms that have not been searched yet ***
    cursor.execute("SELECT term FROM SearchTerms WHERE search_started = 0")
    terms_to_search = [row[0] for row in cursor.fetchall()]
    
    if not terms_to_search:
        print("All search terms have already been processed. No new searches to perform.")
        return

    for term in terms_to_search:
        print(f"\n__________ Searching for term: '{term}' _________")
        
        try:
            search_response = youtube.search().list(
                q=term,
                part="snippet",
                type="video",
                maxResults=MAX_RESULTS_PER_SEARCH,
                relevanceLanguage='en',

            ).execute()
            
            video_ids = [
                item['id']['videoId'] 
                for item in search_response.get("items", []) 
                if item["id"]["kind"] == "youtube#video"
            ]
            
            for video_id in video_ids:
                cursor.execute("SELECT search_term FROM Videos WHERE video_id = ?", (video_id,))
                existing_record = cursor.fetchone()

                if existing_record:
                    # Video exists. Update its search_term.
                    print(f"   [UPDATING] Video {video_id} already in database.")
                    current_terms = existing_record[0]
                    # Use a set to handle existing terms and prevent duplicates
                    term_set = set(current_terms.split('|'))
                    if term not in term_set:
                        term_set.add(term)
                        new_terms = '|'.join(sorted(list(term_set))) # Sort for consistency
                        cursor.execute("UPDATE Videos SET search_term = ? WHERE video_id = ?", (new_terms, video_id))
                        conn.commit()
                        print(f"      -> Appended search term: '{term}'")
                    else:
                        print(f"      -> Search term '{term}' was already associated.")
                    continue # Skip to the next video_id

                # If the video is new, proceed with fetching details and inserting.
                print(f"   -> Processing new video ID: {video_id}")
                
                video_details = get_video_details(youtube, video_id)
                if not video_details: continue

                channel_id = video_details['snippet']['channelId']
                
                channel_data = get_channel_data(youtube, channel_id)

                insert_video(conn, video_details, term)
                insert_channel(conn, channel_data)

                print(f"     [SAVED] Video '{video_details['snippet']['title']}' and its channel.")
                
                time.sleep(0.1)
            
            cursor.execute("UPDATE SearchTerms SET search_started = 1 WHERE term = ?", (term,))
            conn.commit()
            print(f"__________ Marked term '{term}' as complete __________")
        except HttpError as e:
            print(f"An API error occurred for term '{term}': {e}")
            if 'quotaExceeded' in str(e):
                print("Quota exceeded. Stopping the script.")
                break # Exit the loop if quota is exhausted
            
    # Final count check
    video_count = cursor.execute("SELECT COUNT(*) FROM Videos").fetchone()[0]
    channel_count = cursor.execute("SELECT COUNT(*) FROM Channels").fetchone()[0]
    
    print("\n--- DATABASE SUMMARY ---")
    print(f"Videos Stored: {video_count}")
    print(f"Channels Stored: {channel_count}")

# --- EXECUTION ---

if __name__ == '__main__':
    youtube = authenticate_youtube_service()
    conn = sqlite3.connect(DB_NAME)
    
    create_db_tables(conn)
    populate_search_terms(conn, terms)
    update_youtube_database(youtube, conn)
    conn.close()
    print("\n--- Core YouTube Data Collection Finished ---")

    # --- Flag credible channels ---
    flag_credible_channels(db_path=DB_NAME, excel_path=CREDIBILITY_MASTER_FILE)
    
    # --- Create the joined credible videos table ---
    create_credible_videos_table(db_path=DB_NAME)

    # --- Fetch Captions for Credible Videos ---
    fetch_and_save_captions(db_path=DB_NAME)
    
    # --- Score the Captioned Videos ---
    primary_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0, google_api_key=API_KEY, rate_limiter=InMemoryRateLimiter(requests_per_second=66, check_every_n_seconds=0.1, max_bucket_size=5))
        
    # Example: Define configurations if you plan to use different models per section
    model_configurations = {
        0: {'id': 'gemini-2.0-flash-lite', 'llm_instance': primary_llm},
        1: {'id': 'gemini-2.0-flash', 'llm_instance': ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=API_KEY, rate_limiter=InMemoryRateLimiter(requests_per_second=33, check_every_n_seconds=0.1, max_bucket_size=5))},
        2: {'id': 'gemini-2.5-flash-lite', 'llm_instance': ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, google_api_key=API_KEY, rate_limiter=InMemoryRateLimiter(requests_per_second=66, check_every_n_seconds=0.1, max_bucket_size=5))},
        3: {'id': 'gemini-2.5-flash', 'llm_instance': ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=API_KEY, rate_limiter=InMemoryRateLimiter(requests_per_second=16, check_every_n_seconds=0.1, max_bucket_size=5))},
    }
    
    score_videos(db_path=DB_NAME, llm=primary_llm, num_sections=4, model_configs=model_configurations)
    
    print("\n--- Entire Pipeline Finished ---")