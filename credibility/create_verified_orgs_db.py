"""
Create Verified Organizations Database

Author: Zihan Liu
Purpose: Convert Excel master file to SQLite database

Database Schema:
- organization: Organization name
- youtube_channel: YouTube handle
- rank: Credibility/volume rank (NULL for now)
"""

import sqlite3
import pandas as pd


def create_verified_orgs_database(excel_path, db_path, overwrite=True):
    df = pd.read_excel(excel_path)
    print(f"Loaded {len(df)} organizations")

    db_df = df[['Organization', 'YouTube_Channel']].copy()
    db_df.columns = ['organization', 'youtube_channel']
    db_df['rank'] = None

    conn = sqlite3.connect(db_path)

    if_exists = 'replace' if overwrite else 'append'
    db_df.to_sql('verified_organizations', conn, if_exists=if_exists, index=False)

    cursor = conn.cursor()
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_youtube_channel
        ON verified_organizations(youtube_channel)
    """)
    conn.commit()

    print(f"Database created: {db_path}")
    print(f"Table: verified_organizations ({len(db_df)} rows)")

    sample = pd.read_sql_query("SELECT * FROM verified_organizations LIMIT 5", conn)
    print("\nSample data:")
    print(sample.to_string(index=False))

    conn.close()


def update_rank_column(db_path, rank_data):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for org_name, rank in rank_data.items():
        cursor.execute("""
            UPDATE verified_organizations
            SET rank = ?
            WHERE organization = ?
        """, (rank, org_name))

    conn.commit()
    conn.close()
    print(f"Updated rank for {len(rank_data)} organizations")


if __name__ == "__main__":
    EXCEL_FILE = "data/Transplants_Team_Master_File.xlsx"
    DB_FILE = "verified_organizations.db"

    create_verified_orgs_database(
        excel_path=EXCEL_FILE,
        db_path=DB_FILE,
        overwrite=True
    )
