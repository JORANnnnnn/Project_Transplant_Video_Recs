# scoring.py

import sqlite3
import pandas as pd
from typing import List, Tuple
import time
import random

# Assuming your evaluation scripts and the 'langchain' model are accessible
from langchain_google_genai import ChatGoogleGenerativeAI
from Evaluation.HONcodeEvaluate import getHONcodeScore
from Evaluation.DISCERNEvaluate import getDISCERNScore
from Evaluation.PEMATEvaluate import getPEMATScore
from langchain_core.rate_limiters import InMemoryRateLimiter

from pydantic import BaseModel, Field

class UnderstandabilityChecker(BaseModel):
    """Structured output for understandability analysis."""
    analysis: str = Field(description="A concise, single-paragraph analysis (max 100 words) explaining why the transcript is or isn't understandable based on the definition provided.")
    is_understandable: int = Field(ge=0, le=1, description="Score 1 if the transcript is generally understandable according to the definition, 0 otherwise.")

def check_understandability(caption_text: str, llm: ChatGoogleGenerativeAI) -> Tuple[int, str]:
    """
    Uses the provided LLM to assess if the caption text is understandable.

    Args:
        caption_text (str): The video transcript text.
        llm (ChatGoogleGenerativeAI): The Langchain LLM instance (potentially section-specific).

    Returns:
        Tuple[int, str]: A tuple containing the understandability score (0 or 1)
                         and the analysis text. Returns (0, "Error during analysis") on failure.
    """
    try:
        # Define understandability for the LLM
        understandability_definition = (
            "Understandability refers to the quality of information that enables users to process its meaning successfully. "
            "Assess the transcript based on: \n"
            "1. Use of common, everyday language.\n"
            "2. Clear definition of necessary medical or technical terms.\n"
            "3. Logical flow and organization of information.\n"
            "4. Avoidance of excessive jargon or overly complex sentence structures."
        )

        prompt = (
            f"Review the following video transcript related to health topics. "
            f"Based on the definition provided, determine if the transcript is generally understandable for a lay audience. "
            f"Definition of Understandability: {understandability_definition}\n\n"
            f"Provide a score (1 for understandable, 0 for not understandable) and a brief, one-paragraph analysis explaining your reasoning.\n\n"
            f"Transcript:\n{caption_text}"
        )

        structured_llm = llm.with_structured_output(UnderstandabilityChecker)
        result = structured_llm.invoke(prompt)

        # Limit analysis length (optional, but good practice)
        analysis_text = result.analysis[:250] + "..." if len(result.analysis) > 250 else result.analysis

        return result.is_understandable, analysis_text

    except Exception as e:
        print(f"      -> ERROR during understandability check: {e}")
        return 0, f"Error during analysis: {e}"

def _get_labeled_table_schema(conn) -> List[str]:
    """Gets the schema from CaptionedVideos and adds scoring columns."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(CaptionedVideos)")
    columns_info = cursor.fetchall()
    base_columns = [f"{col[1]} {col[2]}" + (" PRIMARY KEY" if col[5] else "") + (" NOT NULL" if col[3] else "") for col in columns_info]

    base_columns = [col for col in base_columns if not col.startswith('HONcode_') and not col.startswith('DISCERN_') and not col.startswith('PEMAT_')]

    # --- UPDATED: Add understandability columns ---
    base_columns.extend([
        "is_understandable INTEGER",
        "understandability_analysis TEXT"
    ])
    # Add other scoring columns
    base_columns.extend([f"HONcode_{i} INTEGER" for i in range(1, 9)])
    base_columns.append("HONcode_score INTEGER")
    base_columns.extend([f"DISCERN_{i} INTEGER" for i in range(1, 16)])
    base_columns.append("DISCERN_score INTEGER")
    base_columns.extend([f"PEMAT_{i} INTEGER" for i in range(1, 14)])
    base_columns.append("PEMAT_score INTEGER")
    base_columns.append("scoring_model_id TEXT")

    return base_columns

def score_videos(db_path: str, llm: ChatGoogleGenerativeAI, num_sections: int = 1, model_configs: dict = None):
    """
    Scores videos from CaptionedVideos using evaluation frameworks and saves to LabeledVideos.
    Includes understandability check.
    """
    print("\n--- Starting Video Scoring Process ---")
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # --- FIX: Move CREATE TABLE to the beginning ---
        # This ensures the LabeledVideos table always exists before we check its columns.
        schema_cols = _get_labeled_table_schema(conn)
        cursor.execute(f"CREATE TABLE IF NOT EXISTS LabeledVideos ({', '.join(schema_cols)});")
        
        # Now, check if the columns need to be added (for runs after the first)
        cursor.execute("PRAGMA table_info(LabeledVideos)")
        existing_cols = [col[1] for col in cursor.fetchall()]
        if 'is_understandable' not in existing_cols:
             print("   -> Adding 'is_understandable' column to LabeledVideos.")
             cursor.execute("ALTER TABLE LabeledVideos ADD COLUMN is_understandable INTEGER;")
        if 'understandability_analysis' not in existing_cols:
             print("   -> Adding 'understandability_analysis' column to LabeledVideos.")
             cursor.execute("ALTER TABLE LabeledVideos ADD COLUMN understandability_analysis TEXT;")
        conn.commit()
        # --- End of Fix ---

        # The rest of the function remains the same...
        query_unscored_videos = """
        SELECT CapV.* FROM CaptionedVideos AS CapV
        LEFT JOIN LabeledVideos AS LabV ON CapV.video_id = LabV.video_id
        WHERE CapV.has_transcript = 1 AND (LabV.video_id IS NULL OR LabV.is_understandable IS NULL);
        """
        videos_to_score_df = pd.read_sql_query(query_unscored_videos, conn)
        
        if videos_to_score_df.empty:
            print("No new captioned videos found needing scoring or understandability check.")
            return

        print(f"Found {len(videos_to_score_df)} videos needing scoring/understandability check.")

        videos_to_score_df = videos_to_score_df.sample(frac=1).reset_index(drop=True)
        if num_sections <= 0: num_sections = 1
        section_size = len(videos_to_score_df) // num_sections
        video_sections = []
        for i in range(num_sections):
            start = i * section_size
            end = (i + 1) * section_size if i < num_sections - 1 else len(videos_to_score_df)
            video_sections.append(videos_to_score_df[start:end])

        print(f"Split videos into {num_sections} sections.")

        total_processed_count = 0
        for section_index, section_df in enumerate(video_sections):
            print(f"\n--- Processing Section {section_index + 1}/{num_sections} ({len(section_df)} videos) ---")

            current_llm = llm
            model_id = f"default_model_{section_index}"
            if model_configs and section_index in model_configs:
                current_llm = model_configs[section_index]['llm_instance']
                model_id = model_configs[section_index]['id']

            print(f"Using model ID: {model_id}")

            for index, row in section_df.iterrows():
                video_id = row['video_id']
                caption_text = row['caption_text']

                if not isinstance(caption_text, str) or len(caption_text) < 10:
                     print(f"   -> Skipping video {video_id} due to invalid or very short caption.")
                     continue

                print(f"   Scoring video {total_processed_count + 1}/{len(videos_to_score_df)}: {video_id}")

                video_record = row.to_dict()
                video_record['scoring_model_id'] = model_id

                try:
                    # --- NEW: Perform Understandability Check FIRST ---
                    print("      -> Checking understandability...")
                    understandable_score, understandability_analysis = check_understandability(caption_text, current_llm)
                    video_record['is_understandable'] = understandable_score
                    video_record['understandability_analysis'] = understandability_analysis
                    # --- End Understandability Check ---

                    # Run Evaluation Frameworks
                    print("      -> Scoring with HONcode...")
                    hon_score, hon_analysis = getHONcodeScore(caption_text, current_llm)

                    print("      -> Scoring with DISCERN...")
                    discern_score, discern_analysis = getDISCERNScore(caption_text, current_llm)

                    print("      -> Scoring with PEMAT...")
                    pemat_score, pemat_analysis = getPEMATScore(caption_text, current_llm)

                    # Update record with scores
                    video_record.update({'HONcode_score': hon_score, 'DISCERN_score': discern_score, 'PEMAT_score': pemat_score})
                    for i, (score, _) in hon_analysis.items(): video_record[f'HONcode_{i}'] = score
                    for i, (score, _) in discern_analysis.items(): video_record[f'DISCERN_{i}'] = score
                    for i, (score, _) in pemat_analysis.items(): video_record[f'PEMAT_{i}'] = score

                    # Use INSERT OR REPLACE
                    # Ensure all potential columns defined in the schema are present in the record
                    # or handle missing keys appropriately before generating SQL
                    schema_col_names = [col.split(' ')[0].strip('"') for col in _get_labeled_table_schema(conn)]
                    final_record_values = []
                    for col_name in schema_col_names:
                        final_record_values.append(video_record.get(col_name, None)) # Use None for missing values

                    cols_sql = ', '.join(f'"{k}"' for k in schema_col_names)
                    placeholders = ', '.join(['?' for _ in schema_col_names])
                    upsert_sql = f"INSERT OR REPLACE INTO LabeledVideos ({cols_sql}) VALUES ({placeholders})"

                    cursor.execute(upsert_sql, final_record_values)
                    conn.commit()
                    total_processed_count += 1

                except Exception as score_err:
                    print(f"      -> ERROR scoring or saving video {video_id}: {score_err}")
                    conn.rollback()

                time.sleep(0.5)

        print(f"\nSuccessfully scored and saved {total_processed_count} videos.")

    except sqlite3.Error as e:
        print(f"A database error occurred during scoring: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during scoring: {e}")
    finally:
        if conn:
            conn.close()
        print("--- Video Scoring Process Finished ---")


if __name__ == '__main__':
    DB_FILE = "youtube_data.db"
    API_KEY = "AIzaSyAK-PTFjHh6qz_Xwl7shmqhafjAtUp_JwM"

    if API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not API_KEY:
        print("Please replace 'YOUR_GEMINI_API_KEY_HERE' with your actual Gemini API key.")
    else:
        primary_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0, google_api_key=API_KEY, rate_limiter=InMemoryRateLimiter(requests_per_second=66, check_every_n_seconds=0.1, max_bucket_size=5))
        
        # Example: Define configurations if you plan to use different models per section
        model_configurations = {
            0: {'id': 'gemini-2.0-flash-lite', 'llm_instance': primary_llm},
            1: {'id': 'gemini-2.0-flash', 'llm_instance': ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=API_KEY, rate_limiter=InMemoryRateLimiter(requests_per_second=33, check_every_n_seconds=0.1, max_bucket_size=5))},
            2: {'id': 'gemini-2.5-flash-lite', 'llm_instance': ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, google_api_key=API_KEY, rate_limiter=InMemoryRateLimiter(requests_per_second=66, check_every_n_seconds=0.1, max_bucket_size=5))},
            3: {'id': 'gemini-2.5-flash', 'llm_instance': ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=API_KEY, rate_limiter=InMemoryRateLimiter(requests_per_second=16, check_every_n_seconds=0.1, max_bucket_size=5))},
        }
        
        score_videos(db_path=DB_FILE, llm=primary_llm, num_sections=4, model_configs=model_configurations)