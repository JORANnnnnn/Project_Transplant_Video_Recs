
import sqlite3
import pandas as pd

def _normalize_handle(handle):
    """Internal helper to normalize YouTube handles for matching."""
    if pd.isna(handle):
        return None
    return str(handle).lower().replace('@', '').strip()

def flag_credible_channels(db_path: str, excel_path: str):
    """
    Adds and updates an 'is_credible' flag in the Channels table.

    This function performs the following steps:
    1. Ensures the 'is_credible' column exists in the Channels table.
    2. Reads the list of verified organizations from the provided Excel file.
    3. Fetches all channels from the main YouTube database.
    4. Matches channels by their normalized YouTube handle.
    5. Updates the 'is_credible' flag to 1 for all matched channels.

    Args:
        db_path (str): The file path to the main youtube_data.db SQLite database.
        excel_path (str): The file path to the Excel master file of verified orgs.
    """
    print("\n--- Starting Credibility Flagging Process ---")
    
    # Load verified organization handles from the Excel file
    try:
        master_df = pd.read_excel(excel_path)
        # Create a set of normalized handles for fast lookups
        verified_handles = set(master_df['YouTube_Channel'].apply(_normalize_handle).dropna())
        print(f"Loaded {len(verified_handles)} unique, verified handles from '{excel_path}'.")
    except Exception as e:
        print(f"ERROR: Could not read or process the Excel file: {e}")
        return

    conn = None
    try:
        # Connect to the main database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Ensure the 'is_credible' column exists in the Channels table
        cursor.execute("PRAGMA table_info(Channels)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'is_credible' not in columns:
            print("Column 'is_credible' not found. Adding it to the Channels table.")
            cursor.execute("ALTER TABLE Channels ADD COLUMN is_credible INTEGER DEFAULT 0")
            conn.commit()

        # Fetch all channels currently in the database
        channels_df = pd.read_sql_query("SELECT channel_id, handle FROM Channels", conn)
        print(f"Found {len(channels_df)} total channels in '{db_path}'.")
        
        # Identify which channels are credible by matching handles
        channels_df['handle_normalized'] = channels_df['handle'].apply(_normalize_handle)
        
        credible_channels = channels_df[channels_df['handle_normalized'].isin(verified_handles)]
        
        if credible_channels.empty:
            print("No channels in the database matched the verified handles list.")
            return

        # Prepare data for the UPDATE statement
        credible_channel_ids = [(channel_id,) for channel_id in credible_channels['channel_id']]

        # Update the database, setting the flag for credible channels
        # First, reset all flags to 0 to handle any potential changes in the master file
        cursor.execute("UPDATE Channels SET is_credible = 0")
        
        # Now, set the flag to 1 for the credible channels
        update_query = "UPDATE Channels SET is_credible = 1 WHERE channel_id = ?"
        cursor.executemany(update_query, credible_channel_ids)
        conn.commit()
        
        print(f"Successfully flagged {cursor.rowcount} channels as credible.")

    except Exception as e:
        print(f"An error occurred during the database update: {e}")
    finally:
        if conn:
            conn.close()
        print("--- Credibility Flagging Process Finished ---")


if __name__ == '__main__':
    # This allows the script to be run directly for a one-time update or testing
    DB_FILE = "youtube_data.db"
    EXCEL_FILE = "Transplants_Team_Master_File.xlsx"
    
    flag_credible_channels(db_path=DB_FILE, excel_path=EXCEL_FILE)