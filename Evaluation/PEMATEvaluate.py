import io
import os
import sqlite3
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain_core.rate_limiters import InMemoryRateLimiter
import pandas as pd
import csv
import time
from typing import List, Tuple

#my_list = ['Nlt5PBrowKE', '4CCA9ojRAoU', 'vj-8ikMXPPk', '0H_xTEHRK0g', 'kOFVP0dPl-Q', 'tDrKOe8cs_0', 'awC12PFFUuM', 'tUZaEuBnycc', 'TelxHyKXGgw', 'Hjnd67MKhy0', '8uLx9drXbkA', 'jIwUwGE9aX8', 'SsXXGYdBdGk', 'X5HxpB8u0h8', 'V_I6Q56vtjE', 'h6l3CB3boLA', 'upTj79dRmZk', 'iEOa6VM8zHg', '1A51_BQpTUU', '9ItbKVrDOtQ', 'YJwkoX7KoOA', 'j163VvTQ3q8', 'w-H1eYdPhPI', 'wApvCj3HDAU', '03jZdOzvjSc', '9_VsEs_T1hA', '2zy9LV6ARSI', 'UOe4tcUhbQI', 'lxVGPACR5ok', 'V3O_FpF4ggo', '7YRFE8tg-gY', 'Q75-R4EMtZI', 'WA2U293-GRg', 'plNUfhcTJnI', 'Oj5N4H6Y9wg', 'uag1jIEuKTw', 'a4ibn9IVChI', '4pixAOy_yrU', 'CaPxqmkBDEg', 'vk3B1EmqhCM', 'GXyEZXJCoYc', 'GJ_g6aDAx9k', 'ShQNrFfEYYs', 'vlkASwAkHaE', 'y9EwhX7uyZ0', '6zP1g_Y_wSo', 'XV-pGEY4rxc', 'rP05zGjuZu8', 'N7kj0sfe6og', 'K69DQUq-HGo']
my_list = ['LZH5EQAfBVY','kGlGGqoF5-o','9_VsEs_T1hA','Q75-R4EMtZI','1A51_BQpTUU','V0yazJUTlb4','jPSl50VPOHI','3Fker8Nmd5Y','fPyS5ma_7rM','m7qMAZ4ALKU','MhXkknPC_vc','HJCRWBLEWAM','uag1jIEuKTw','HggDQUjOnDQ','4pixAOy_yrU','9WAD1YM5R14','U8ZVlKI_yoA','XV-pGEY4rxc','V3O_FpF4ggo','obss0_8xIBs','szMgN68kijg','ojpeeno-ppg','lZl8tISca5E','s4ZVdc9sz2A','Nlt5PBrowKE','ArD4wIAZtxI','03jZdOzvjSc','SsXXGYdBdGk','X5HxpB8u0h8','9ItbKVrDOtQ','wiziIhGjUHA','tDrKOe8cs_0','idX2wd0HY30','JNjH39gHIww','w4iz4OTCySM','A-M6wSyw6IY','E9Dnyym6a94','DjTC8drkMEE','fMuGNy2oCzI','u1pg9QjrNzs','awC12PFFUuM','GeMEopvNJUM','13X23gF0K_4','98qvFKR8OmQ','HGeInTptfyU','0H_xTEHRK0g','KIvYlZ_HaEA','SdH-IhgHD8U','1IauflS1rR0','cWbZbyLmY5Q','_48-eWGtSZ8','iEOa6VM8zHg','w-H1eYdPhPI','1hKkuOSe2vw','j163VvTQ3q8','86Iy9c8rSP0','CaPxqmkBDEg','vk3B1EmqhCM','GXyEZXJCoYc','lvSks9_fu0A','6obZRepfzGY','CqeNdL_qJEg','V_I6Q56vtjE','pdZFceZaiik','OLKfH44FVk0','K69DQUq-HGo','-QtYNPHW0mQ','bvCRH0A6kxY','iWSewFlU6o8','19KrsHv1OQw','oc6vCBmvrzk','1KceELBEzgw','f-Uf0XUMqO0','GHyF9JuWE6Q','aS3XEPPfNtU','7rDt5QGWZf0','84yjo_SqvVA','zqNTr084XTk','iZjpNbYrU4w','oWT8_BOEtRk','rwFOG_rk4Nk','d46WEOlnuVg','jsLa9itfNFU','Ja3qDHvAXnU','DgBYYFwd42A','JS0-opsKPPw','NGr8q36Cmj8','IFLM_lpiS70','2n4gXcZ1CMc','PnnuV7NCzow','cVSSdly_m7A','S3CegUhEd7Y','voEQjWvxBRM','lGktTbcUu3E','1IjYsYT6KmQ','O298tg7_GQs','Svuy0sjpP2A','30O9u8iC51g','ShQNrFfEYYs','y9EwhX7uyZ0','6zP1g_Y_wSo','rP05zGjuZu8','iPyVe4jx7rI','fKyzKlYGZhA','o1XNQs21gGA','krjuiYAzBkk','VpSQoQK_Wek','14m0iXpHVAU','xUV0eKlDJcY','VOfqGYNxFs4','jRivbMime1s','zgb8yOYYc68','24v2zQHZUAE','GJ_g6aDAx9k','vlkASwAkHaE','a4ibn9IVChI','8uLx9drXbkA','plNUfhcTJnI','lxVGPACR5ok','kOFVP0dPl-Q','YJwkoX7KoOA','h6l3CB3boLA','4CCA9ojRAoU','Q88c7drpQrw','ohGLSYxHFzg','jIwUwGE9aX8','mPwX8P6EL4s','gOLAVmFflgg','Xa-iEHml8nc','TelxHyKXGgw','Hjnd67MKhy0','Ra_OdIPXdqE','rVyQbCtRKeA','Sr4ybaI4y6M','VGZRMHA4ics','N7kj0sfe6og','2zy9LV6ARSI','SuHkwjqlGEg','2WOM6we0SyM','0UvOSHJz7U0','sYDMzXFdU0g','kh2Nb0huvQs','Ca353BNz0xU','QPj6e1aqiB0','pn7x1GLjyoE','LpiytTB4ZG4','vqeDdme8V08','OiRnjmQUylQ','RLz_PVzRb8o','rrajxcaj5bw','4UUyBAeVmQk','WZxopWCd9C0','OkHESl_I2O8','gt-AVLJGYnA','a7gJ73YIz40','4tTwjfZs3u8','kv3PN9sti9Y','9tt-HuJRIIQ','PLDHbo9nQGY','IrgOE0B8xIw','vXxVFIgNTCE','1PtxaxcPnGc','4L4FrsM6kFM','JY8Dp1pKx1U','ZCpSiSS0Yv4','3aUEf7VZLas','1G5Kkl_xjFY','oBzWEtWY--0','i1Tb5gfymfU','AUMTTZ5aVIQ','vj-8ikMXPPk','qkWLnyHua1g','8ufoR2cQ04g','ofLDgqgnzlg','UOe4tcUhbQI','MeSYFx7ends','WA2U293-GRg','QapEAECWcJY','_kwtFLe2VQQ','2uD0VmPdGIE','Ke-PedBEhpk','IkBMQ2QTmfc','aKmcTOY4N0M','Lp1eghw7_7Q','ZXqnSAxv99M']
my_set = set(my_list[50:])

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

def getPEMATScore(caption : str, llm : ChatGoogleGenerativeAI):
    PEMAT = {
        1 : ("The material makes its purpose completely evident.", 1),
        2 : ("The material uses common, everyday language.", 1),
        3 : ("Medical terms are used only to familiarize audience with the terms. When used, medical terms are defined.", 1),
        4 : ("The material uses the active voice.", 1),
        5 : ('The material breaks or "chunks" information into short sections.', 1),
        6 : ("The material’s sections have informative headers.", 0),
        7 : ("The material presents information in a logical sequence.", 1),
        8 : ("The material provides a summary.", 1),
        9 : ("The material uses visual cues (e.g., arrows, boxes, bullets, bold, larger font, highlighting) to draw attention to key points.", 0),
        10 : ("Text on the screen is easy to read.", 0),
        11 : ("The material allows the user to hear the words clearly (e.g., not too fast, not garbled).", 1),
        12 : ("The material uses illustrations and photographs that are clear and uncluttered.", 0),
        13 : ("The material uses simple tables with short and clear row and column headings.", 0)
    }

    handbook_path = "PEMAT Guide.txt"
    loader = TextLoader(handbook_path)
    docs = loader.load()
    handbook_document = docs[0]
    handbook = handbook_document.page_content

    class PEMATScorer(BaseModel):
        """Always use this tool to structure your response to the user."""
        analysis: str = Field(description="The answer to the user's question")
        score: int = Field(ge=0, le=1, description="Give it a compliance: 1 if the caption is compliant with the principle, 0 otherwise.")

    structured_llm = llm.with_structured_output(PEMATScorer)

    analysis = {}
    total_score = 0
    for count, (principle, valid) in PEMAT.items():
        if valid == 1:
            print(f"   -> Evaluating principle {count}/8.")
            system_message = SystemMessage(content=f"Use the following PEMAT guide to analyze and rate each caption against the guidelines of the item.\n\n{handbook}")
            prompt = f"Consider the captions attached from a YouTube video related to organ transplant. Analyze if the caption comply with the following principle:\n\n{principle}\n\nGive it a compliance score 1 if compliant, 0 otherwise.\n\nCaptions:\n\n{caption}"
            human_message = HumanMessage(content=prompt)
            messages = [system_message, human_message]
            result = structured_llm.invoke(messages)
            analysis[count] = (result.score, result.analysis)
            total_score += result.score

    return total_score, analysis

if __name__ == '__main__':
    DB_FILE = "youtube_data.db"
    API_KEY = "AIzaSyAqEQOjFDxdMrUKuTcyA_3mbnZ0s9bL500"

    if API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not API_KEY:
        print("Please replace 'YOUR_GEMINI_API_KEY_HERE' with your actual Gemini API key.")
    else:
        primary_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0, google_api_key=API_KEY, rate_limiter=InMemoryRateLimiter(requests_per_second=2, check_every_n_seconds=0.1, max_bucket_size=1))
    
    print("\n--- Starting Video Scoring Process ---")
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        query_unscored_videos = """
        SELECT CapV.* FROM CaptionedVideos AS CapV
        LEFT JOIN LabeledVideos AS LabV ON CapV.video_id = LabV.video_id
        WHERE CapV.has_transcript = 1 AND LabV.is_understandable IS NOT NULL;
        """
        videos_to_score_df = pd.read_sql_query(query_unscored_videos, conn)
        videos_to_score_df = videos_to_score_df[videos_to_score_df['video_id'].isin(my_set)].reset_index(drop=True)
    except sqlite3.Error as e:
        print(f"A database error occurred during scoring: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during scoring: {e}")
    finally:
        if conn:
            conn.close()
        print("--- Video Scoring Process Finished ---")

    total_processed_count = 0
    processed_results = []
    for index, row in videos_to_score_df.iterrows():
        video_id = row['video_id']
        caption_text = row['caption_text']

        if not isinstance(caption_text, str) or len(caption_text) < 10:
            print(f"   -> Skipping video {video_id} due to invalid or very short caption.")
            continue

        print(f"   Scoring video {total_processed_count + 1}/{len(videos_to_score_df)}: {video_id}")

        video_record = row.to_dict()
        video_record['scoring_model_id'] = "gemini-2.5-pro"

        try:
            # --- NEW: Perform Understandability Check FIRST ---
            print("      -> Checking understandability...")
            understandable_score, understandability_analysis = check_understandability(caption_text, primary_llm)
            video_record['is_understandable'] = understandable_score
            video_record['understandability_analysis'] = understandability_analysis
            # --- End Understandability Check ---

            # Run Evaluation Frameworks
            # print("      -> Scoring with HONcode...")
            # hon_score, hon_analysis = getHONcodeScore(caption_text, current_llm)

            # print("      -> Scoring with DISCERN...")
            # discern_score, discern_analysis = getDISCERNScore(caption_text, current_llm)

            print("      -> Scoring with PEMAT...")
            pemat_score, pemat_analysis = getPEMATScore(caption_text, primary_llm)

            # Update record with scores
            video_record.update({'PEMAT_score': pemat_score, 'PEMAT_analysis': pemat_analysis})
            # for i, (score, _) in hon_analysis.items(): video_record[f'HONcode_{i}'] = score
            # for i, (score, _) in discern_analysis.items(): video_record[f'DISCERN_{i}'] = score
            for i, (score, _) in pemat_analysis.items(): video_record[f'PEMAT_{i}'] = score

            # Use INSERT OR REPLACE
            # Ensure all potential columns defined in the schema are present in the record
            # or handle missing keys appropriately before generating SQL
            # schema_col_names = [col.split(' ')[0].strip('"') for col in _get_labeled_table_schema(conn)]
            # final_record_values = []
            # for col_name in schema_col_names:
            #     final_record_values.append(video_record.get(col_name, None)) # Use None for missing values

            # cols_sql = ', '.join(f'"{k}"' for k in schema_col_names)
            # placeholders = ', '.join(['?' for _ in schema_col_names])
            # upsert_sql = f"INSERT OR REPLACE INTO LabeledVideos ({cols_sql}) VALUES ({placeholders})"

            # cursor.execute(upsert_sql, final_record_values)
            # conn.commit()
            # fieldnames = list(video_record.keys())
            # with open('first_50_videos.csv', 'w', newline='') as csvfile:
            #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            #     writer.writeheader()
            #     writer.writerow(video_record)
            processed_results.append(video_record)
            total_processed_count += 1

        except Exception as score_err:
            print(f"      -> ERROR scoring or saving video {video_id}: {score_err}")

    if processed_results:
        print(f"Saving {len(processed_results)} videos to CSV...")
        results_df = pd.DataFrame(processed_results)
        results_df.to_csv('first_50_videos.csv', index=False)
    else:
        print("No videos were processed successfully.")

    print(f"\nSuccessfully scored and saved {total_processed_count} videos.")