# enrich_credible_videos.py

import sqlite3
import pandas as pd
from typing import Tuple, List
import time

# Assuming your evaluation scripts and the 'langchain' model are accessible
from langchain_google_genai import ChatGoogleGenerativeAI
from Evaluation.HONcodeEvaluate import getHONcodeScore
from Evaluation.DISCERNEvaluate import getDISCERNScore
from Evaluation.PEMATEvaluate import getPEMATScore

def try_fetch_transcript(video_id: str, languages: Tuple[str, ...] = ("en", "en-US", "en-GB")) -> Tuple[bool, str]:
    """
    Works with both legacy (<1.x) and new (>=1.x) youtube-transcript-api.
    Returns (ok, text_or_error).
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
    except Exception as e:
        return (False, f"{e}")

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
        return (True, text) if text else (False, "Transcript fetched but was empty.")
    except (NoTranscriptFound, TranscriptsDisabled):
        return (False, "No transcript found or transcripts are disabled for this video.")
    except Exception as e:
        if "blocked" in str(e).lower():
            return (False, f"YouTube is blocking requests (likely IP ban): {e}")
        return (False, f"An unexpected error occurred while fetching transcript: {e}")

def _get_table_schema_columns() -> List[str]:
    """Returns the list of column definitions for the final scores table."""
    columns_and_types = [
        "video_id TEXT PRIMARY KEY NOT NULL", "channel_id TEXT NOT NULL", "search_term TEXT",
        "title TEXT", "description TEXT", "published_at TEXT", "duration TEXT",
        "view_count INTEGER", "like_count INTEGER", "comment_count INTEGER", "handle TEXT",
        "channel_title TEXT", "subscriber_count INTEGER", "is_credible INTEGER",
        "video_count INTEGER", "channel_view_count INTEGER",
        "caption_text TEXT", "has_transcript INTEGER DEFAULT 0"
    ]
    # Add evaluation framework columns
    columns_and_types.extend([f"HONcode_{i} INTEGER" for i in range(1, 9)])
    columns_and_types.append("HONcode_score INTEGER")
    columns_and_types.extend([f"DISCERN_{i} INTEGER" for i in range(1, 16)])
    columns_and_types.append("DISCERN_score INTEGER")
    columns_and_types.extend([f"PEMAT_{i} INTEGER" for i in range(1, 14)])
    columns_and_types.append("PEMAT_score INTEGER")
    return columns_and_types

def enrich_and_score_videos(db_path: str, llm: ChatGoogleGenerativeAI):
    """
    Enriches the CredibleVideos table with captions and credibility scores,
    and supports incremental processing.
    """
    print("\n--- Starting Video Enrichment and Scoring Process ---")
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 1. Ensure the target table schema exists with the new columns
        schema_cols = _get_table_schema_columns()
        cursor.execute(f"CREATE TABLE IF NOT EXISTS CredibleVideosWithScores ({', '.join(schema_cols)});")
        conn.commit()
        
        # 2. Identify videos to process
        query_unprocessed_videos = """
        SELECT CV.* FROM CredibleVideos AS CV
        LEFT JOIN CredibleVideosWithScores AS CVS ON CV.video_id = CVS.video_id
        WHERE CVS.video_id IS NULL OR CVS.has_transcript = 0;
        """
        videos_to_process_df = pd.read_sql_query(query_unprocessed_videos, conn)
        
        if videos_to_process_df.empty:
            print("No new or unprocessed credible videos found. All done!")
            return

        print(f"Found {len(videos_to_process_df)} new or unprocessed credible videos.")

        # --- Drop unwanted columns from the source DataFrame ---
        cols_to_drop = [col for col in videos_to_process_df.columns if 'HONcode_' in col or 'DISCERN_' in col]
        if cols_to_drop:
            videos_to_process_df.drop(columns=cols_to_drop, inplace=True)
            print(f"   -> Dropped legacy columns from source data: {', '.join(cols_to_drop)}")

        # 3. Process each video
        for index, row in videos_to_process_df.iterrows():
            video_id = row['video_id']
            print(f"Processing video {index + 1}/{len(videos_to_process_df)}: {video_id}")
            
            ok, fetch_result = try_fetch_transcript(video_id)
            
            video_record = row.to_dict()
            video_record['has_transcript'] = 1 if ok else 0
            video_record['caption_text'] = fetch_result

            if ok:
                print(f"   -> Caption found. Length: {len(fetch_result)} chars.")
                # --- Run Evaluation Frameworks ---
                print("   -> Scoring with HONcode...")
                hon_score, hon_analysis = getHONcodeScore(fetch_result, llm)
                
                print("   -> Scoring with DISCERN...")
                discern_score, discern_analysis = getDISCERNScore(fetch_result, llm)

                print("   -> Scoring with PEMAT...")
                pemat_score, pemat_analysis = getPEMATScore(fetch_result, llm)
            else:
                print(f"   -> Failed to fetch caption: {fetch_result}")
                hon_score, hon_analysis = 0, {i: (None, "") for i in range(1, 9)}
                discern_score, discern_analysis = 0, {i: (None, "") for i in range(1, 16)}
                pemat_score, pemat_analysis = 0, {i: (None, "") for i in range(1, 14)}

            # Update record with scores
            video_record.update({'HONcode_score': hon_score, 'DISCERN_score': discern_score, 'PEMAT_score': pemat_score})
            for i, (score, _) in hon_analysis.items(): video_record[f'HONcode_{i}'] = score
            for i, (score, _) in discern_analysis.items(): video_record[f'DISCERN_{i}'] = score
            for i, (score, _) in pemat_analysis.items(): video_record[f'PEMAT_{i}'] = score

            # 4. Use INSERT OR REPLACE to upsert the record
            cols = ', '.join(video_record.keys())
            placeholders = ', '.join(['?' for _ in video_record])
            upsert_sql = f"INSERT OR REPLACE INTO CredibleVideosWithScores ({cols}) VALUES ({placeholders})"
            cursor.execute(upsert_sql, list(video_record.values()))
            conn.commit()

            # --- Check for IP block and break the loop if detected ---
            if not ok and "blocking requests" in fetch_result.lower():
                print("\nIP block detected. Halting the enrichment process.")
                print("The failed video's status has been saved. You can rerun the script later.")
                break  # Exit the for loop
            
            time.sleep(0.5)

        print(f"\nSuccessfully processed and updated/inserted {len(videos_to_process_df)} videos.")

    except Exception as e:
        print(f"An error occurred during the enrichment process: {e}")
    finally:
        if conn:
            conn.close()
        print("--- Enrichment and Scoring Process Finished ---")

if __name__ == '__main__':
    DB_FILE = "youtube_data.db"
    API_KEY = "AIzaSyC7tjdbVPnPUxwsPA7WLTnEywh10G1Ry2M"

    if API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not API_KEY:
        print("Please replace 'YOUR_GEMINI_API_KEY_HERE' with your actual Gemini API key.")
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, google_api_key=API_KEY)
        enrich_and_score_videos(db_path=DB_FILE, llm=llm)