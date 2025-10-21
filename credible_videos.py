
import sqlite3
import pandas as pd

def create_credible_videos_table(db_path: str):
    """
    Creates a new table named 'CredibleVideos' by joining the Videos and
    Channels tables, filtered for channels flagged as credible.

    This function will:
    1. Connect to the main YouTube database.
    2. Perform an INNER JOIN between the Videos and Channels tables on 'channel_id'.
    3. Filter the results to include only rows where 'is_credible' is 1.
    4. Create a new table named 'CredibleVideos' with the combined data,
       replacing it if it already exists to ensure it's always up-to-date.

    Args:
        db_path (str): The file path to the main youtube_data.db SQLite database.
    """
    print("\n--- Generating Final Credible Videos Table ---")
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        
        # This SQL query joins the two tables and selects all columns,
        # filtering for only the credible channels.
        query = """
        SELECT
            V.*,
            C.handle,
            C.title AS channel_title,
            C.subscriber_count,
            C.video_count,
            C.view_count AS channel_view_count,
            C.is_credible
        FROM
            Videos AS V
        INNER JOIN
            Channels AS C ON V.channel_id = C.channel_id
        WHERE
            C.is_credible = 1;
        """
        
        # Use pandas to execute the query and get the results
        credible_videos_df = pd.read_sql_query(query, conn)
        
        if credible_videos_df.empty:
            print("No videos from credible channels were found. 'CredibleVideos' table will not be created.")
            return

        print(f"Found {len(credible_videos_df)} videos from credible channels.")
        
        # Save the resulting DataFrame to a new table named 'CredibleVideos'.
        # 'if_exists='replace'' ensures the table is fresh every time the script runs.
        credible_videos_df.to_sql('CredibleVideos', conn, if_exists='replace', index=False)
        
        print(f"Successfully created/updated the 'CredibleVideos' table with {len(credible_videos_df)} rows.")

    except Exception as e:
        print(f"An error occurred while creating the credible videos table: {e}")
    finally:
        if conn:
            conn.close()
        print("--- Final Table Generation Finished ---")


if __name__ == '__main__':
    # This allows the script to be run directly for testing or one-off generation
    DB_FILE = "youtube_data.db"
    create_credible_videos_table(db_path=DB_FILE)