# Data Pipeline

Use this piece of code to write the database tables into a Excel file. It would be helpful for working with the videos.

```
import sqlite3
import pandas as pd

DB_NAME = 'youtube_data.db'

# Connect to the database
conn = sqlite3.connect(DB_NAME)

# Load an entire table (e.g., Videos) into a DataFrame
df_videos = pd.read_sql_query("SELECT * FROM videos;", conn)
df_captions = pd.read_sql_query("SELECT * FROM captions;", conn)
df_channels = pd.read_sql_query("SELECT * FROM channels;", conn)

with pd.ExcelWriter('YOUTUBE DB.xlsx') as writer:
    # Write each DataFrame to a separate sheet
    df_videos.to_excel(writer, sheet_name='Videos', index=False)
    df_captions.to_excel(writer, sheet_name='Captions', index=False)
    df_channels.to_excel(writer, sheet_name='Channels', index=False)

```
