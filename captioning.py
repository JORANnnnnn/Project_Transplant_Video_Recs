# captioning.py

import sqlite3
import pandas as pd
from typing import Tuple, List
import time

my_list = ['Nlt5PBrowKE', '4CCA9ojRAoU', 'vj-8ikMXPPk', '0H_xTEHRK0g', 'kOFVP0dPl-Q', 'tDrKOe8cs_0', 'awC12PFFUuM', 'tUZaEuBnycc', 'TelxHyKXGgw', 'Hjnd67MKhy0', '8uLx9drXbkA', 'jIwUwGE9aX8', 'SsXXGYdBdGk', 'X5HxpB8u0h8', 'V_I6Q56vtjE', 'h6l3CB3boLA', 'upTj79dRmZk', 'iEOa6VM8zHg', '1A51_BQpTUU', '9ItbKVrDOtQ', 'YJwkoX7KoOA', 'j163VvTQ3q8', 'w-H1eYdPhPI', 'wApvCj3HDAU', '03jZdOzvjSc', '9_VsEs_T1hA', '2zy9LV6ARSI', 'UOe4tcUhbQI', 'lxVGPACR5ok', 'V3O_FpF4ggo', '7YRFE8tg-gY', 'Q75-R4EMtZI', 'WA2U293-GRg', 'plNUfhcTJnI', 'Oj5N4H6Y9wg', 'uag1jIEuKTw', 'a4ibn9IVChI', '4pixAOy_yrU', 'CaPxqmkBDEg', 'vk3B1EmqhCM', 'GXyEZXJCoYc', 'GJ_g6aDAx9k', 'ShQNrFfEYYs', 'vlkASwAkHaE', 'y9EwhX7uyZ0', '6zP1g_Y_wSo', 'XV-pGEY4rxc', 'rP05zGjuZu8', 'N7kj0sfe6og', 'K69DQUq-HGo']
my_set = set(my_list)

def try_fetch_transcript(video_id: str, languages: Tuple[str, ...] = ("en", "en-US", "en-GB")) -> Tuple[bool, str]:
    """
    Works with both legacy (<1.x) and new (>=1.x) youtube-transcript-api.
    Returns (ok, text_or_error).
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
        from youtube_transcript_api.proxies import WebshareProxyConfig
    except Exception as e:
        return (False, f"{e}")

    try:
        langs = list(languages)
        # Legacy API provided a static method; newer API provides an instance with .fetch()
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            data = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
        else:
            proxy = WebshareProxyConfig(
                proxy_port=80,
                proxy_username="tqykthyb-4",
                proxy_password="p6l8prilr9o8"
            )
            api = YouTubeTranscriptApi(proxy_config=proxy)
            
            #api = YouTubeTranscriptApi()
            fetched = api.fetch(video_id, languages=langs)
            if hasattr(fetched, "to_dict"):
                data = fetched.to_dict()
            elif hasattr(fetched, "to_raw_data"):
                data = fetched.to_raw_data()
            else:
                data = list(fetched)

        text = " ".join(d.get("text", "") for d in data if isinstance(d, dict) and d.get("text")).strip()
        return (True, text) if text else (False, "Transcript fetched but was empty.")
    except (NoTranscriptFound, TranscriptsDisabled):
        return (False, "No transcript found or transcripts are disabled for this video.")
    except Exception as e:
        if "blocked" in str(e).lower():
            return (False, f"YouTube is blocking requests (likely IP ban): {e}")
        return (False, f"An unexpected error occurred while fetching transcript: {e}")

def _get_captioned_table_schema(conn) -> List[str]:
    """Gets the schema from CredibleVideos and adds captioning columns."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(CredibleVideos)")
    columns_info = cursor.fetchall()
    # (id, name, type, notnull, default_value, pk)
    base_columns = [f"{col[1]} {col[2]}" + (" PRIMARY KEY" if col[5] else "") + (" NOT NULL" if col[3] else "") for col in columns_info]
    
    # Remove potential duplicates if they somehow got into CredibleVideos
    base_columns = [col for col in base_columns if not col.startswith('caption_text') and not col.startswith('has_transcript')]

    # Add new caption-specific columns
    base_columns.extend([
        "caption_text TEXT",
        "has_transcript INTEGER DEFAULT 0"
    ])
    return base_columns

def fetch_and_save_captions(db_path: str):
    """
    Fetches captions for videos in CredibleVideos and saves them to CaptionedVideos.
    Supports incremental processing and stops on IP blocks.
    """
    print("\n--- Starting Video Captioning Process ---")
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 1. Ensure the target table schema exists
        schema_cols = _get_captioned_table_schema(conn)
        cursor.execute(f"CREATE TABLE IF NOT EXISTS CaptionedVideos ({', '.join(schema_cols)});")
        conn.commit()
        
        # 2. Identify videos to process (credible videos not yet successfully captioned)
        query_unprocessed_videos = """
        SELECT CV.* FROM CredibleVideos AS CV
        LEFT JOIN CaptionedVideos AS CapV ON CV.video_id = CapV.video_id
        WHERE CapV.video_id IS NULL OR CapV.has_transcript = 0;
        """
        videos_to_process_df = pd.read_sql_query(query_unprocessed_videos, conn)
        videos_to_process_df = videos_to_process_df[videos_to_process_df['video_id'].isin(my_set)].reset_index(drop=True)
        
        if videos_to_process_df.empty:
            print("No new or uncaptioned credible videos found.")
            return

        print(f"Found {len(videos_to_process_df)} credible videos needing captions.")

        # 3. Process each video
        for index, row in videos_to_process_df.iterrows():
            video_id = row['video_id']
            print(f"Processing video {index + 1}/{len(videos_to_process_df)}: {video_id}")
            
            ok, fetch_result = try_fetch_transcript(video_id)
            
            video_record = row.to_dict() # Get data from CredibleVideos
            video_record['has_transcript'] = 1 if ok else 0
            video_record['caption_text'] = fetch_result

            if ok:
                print(f"   -> Caption found. Length: {len(fetch_result)} chars.")
            else:
                print(f"   -> Failed to fetch caption: {fetch_result}")

            # 4. Use INSERT OR REPLACE to upsert the record into CaptionedVideos
            # Ensure keys are quoted in case of special characters
            cols = ', '.join(f'"{k}"' for k in video_record.keys())
            placeholders = ', '.join(['?' for _ in video_record])
            upsert_sql = f"INSERT OR REPLACE INTO CaptionedVideos ({cols}) VALUES ({placeholders})"
            
            try:
                cursor.execute(upsert_sql, list(video_record.values()))
                conn.commit()
            except sqlite3.Error as sql_err:
                print(f"   -> SQL Error during upsert for {video_id}: {sql_err}")
                conn.rollback() # Rollback the failed transaction
                continue # Skip to the next video

            # 5. Check for IP block and break the loop if detected
            if not ok and "blocking requests" in fetch_result.lower():
                print("\nIP block detected during captioning. Halting the process.")
                print("The failed video's status has been saved. You can rerun later.")
                break
            
            time.sleep(0.05) # Small delay between transcript fetches

        processed_count = index + 1 if 'index' in locals() else 0
        print(f"\nSuccessfully processed captions for {processed_count} videos.")

    except sqlite3.Error as e:
        print(f"A database error occurred during captioning: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during captioning: {e}")
    finally:
        if conn:
            conn.close()
        print("--- Video Captioning Process Finished ---")

if __name__ == '__main__':
    # Allows the script to be run directly for testing
    DB_FILE = "youtube_data.db"
    fetch_and_save_captions(db_path=DB_FILE)