# db_cleanup.py

import sqlite3
import pandas as pd

DB_NAME = 'youtube_data.db'
TABLES_TO_CLEAN = ['Videos', 'CredibleVideos']

def clean_intermediate_tables(db_path: str):
    """
    Removes legacy HONcode_ and DISCERN_ columns from intermediate tables.
    This version correctly handles schemas with and without quoted column names.
    """
    print("--- Starting Database Cleanup Process ---")
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        for table_name in TABLES_TO_CLEAN:
            print(f"\nChecking table: '{table_name}'...")
            
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if not cursor.fetchone():
                print(f"   -> Table '{table_name}' not found. Skipping.")
                continue

            cursor.execute(f"PRAGMA table_info({table_name})")
            columns_info = cursor.fetchall()
            current_columns = [col[1] for col in columns_info]
            columns_to_keep = [col for col in current_columns if not ('HONcode_' in col or 'DISCERN_' in col)]

            if len(columns_to_keep) == len(current_columns):
                print(f"   -> No legacy score columns found in '{table_name}'. No action needed.")
                continue
            
            dropped_cols = [col for col in current_columns if col not in columns_to_keep]
            print(f"   -> Found legacy columns to drop: {', '.join(dropped_cols)}")

            temp_table_name = f"{table_name}_temp_clean"
            
            cursor.execute("BEGIN TRANSACTION;")

            # Fetch the original CREATE statement
            df_schema = pd.read_sql(f"SELECT sql FROM sqlite_master WHERE name = '{table_name}'", conn)
            original_create_sql = df_schema['sql'].iloc[0]
            
            # Extract the part between the parentheses
            create_sql_cols_part = original_create_sql.split('(', 1)[1].rsplit(')', 1)[0]
            
            new_cols_defs = []
            for col_def in create_sql_cols_part.split(','):
                # --- THIS IS THE CORRECTED LOGIC ---
                # It now strips quotes to correctly identify the column name
                col_name_raw = col_def.strip().split(' ')[0]
                col_name_clean = col_name_raw.strip('"').strip('`').strip("'") 
                
                if col_name_clean in columns_to_keep:
                    new_cols_defs.append(col_def.strip())
            
            new_create_sql = f"CREATE TABLE {temp_table_name} ({', '.join(new_cols_defs)})"
            
            cursor.execute(new_create_sql)
            
            columns_to_keep_str = ', '.join([f'"{col}"' for col in columns_to_keep])
            cursor.execute(f"INSERT INTO {temp_table_name} ({columns_to_keep_str}) SELECT {columns_to_keep_str} FROM {table_name};")
            
            cursor.execute(f"DROP TABLE {table_name};")
            cursor.execute(f"ALTER TABLE {temp_table_name} RENAME TO {table_name};")
            conn.commit()

            print(f"   -> Successfully cleaned and recreated table '{table_name}'.")

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"\nAn error occurred during the cleanup process: {e}")
    finally:
        if conn:
            conn.close()
        print("\n--- Database Cleanup Process Finished ---")


if __name__ == '__main__':
    # Run the cleanup function directly
    clean_intermediate_tables(db_path=DB_NAME)