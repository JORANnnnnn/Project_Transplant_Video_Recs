# db_utils.py

import sqlite3
import pandas as pd

DB_NAME = 'youtube_data.db'

def print_db_schema(db_path: str = DB_NAME):
    """
    Connects to the SQLite database and prints the schema for all tables.
    """
    print(f"--- Schema for {db_path} ---")
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get a list of all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables found in the database.")
            return
            
        # For each table, print its schema
        for table_name_tuple in tables:
            table_name = table_name_tuple[0]
            print(f"\n[Table: {table_name}]")
            cursor.execute(f"PRAGMA table_info({table_name});")
            schema = cursor.fetchall()
            for column in schema:
                # column format: (id, name, type, notnull, default_value, pk)
                print(f"  - {column[1]} ({column[2]})")
                
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

def export_all_tables_to_excel(db_path: str = DB_NAME, excel_path: str = 'youtube_database_export.xlsx'):
    """
    Exports all tables from the SQLite database to a single multi-sheet Excel file.
    """
    print(f"\n--- Exporting all tables from {db_path} to {excel_path} ---")
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables to export.")
            return

        with pd.ExcelWriter(excel_path) as writer:
            for table_name_tuple in tables:
                table_name = table_name_tuple[0]
                print(f"  -> Exporting table '{table_name}'...")
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                df.to_excel(writer, sheet_name=table_name, index=False)
        
        print(f"\n✅ Successfully exported {len(tables)} tables to '{excel_path}'.")

    except Exception as e:
        print(f"An error occurred during export: {e}")
    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    # You can run this script directly to use its functions
    
    # 1. Print the schema of the database
    print_db_schema()
    
    # 2. Export all tables to an Excel file
    export_all_tables_to_excel()