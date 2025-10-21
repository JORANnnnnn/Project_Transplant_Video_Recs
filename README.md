YouTube Health Content Analysis Pipeline
========================================

This project is an automated data pipeline designed to collect, filter, and evaluate health-related video content from YouTube. It identifies videos from credible sources, fetches their transcripts, and scores them against established health communication frameworks (HONcode, DISCERN, and PEMAT).

The final output is a SQLite database containing a detailed, enriched table of credible videos with their associated scores and metadata.

Project Structure
-----------------

-   `youtube_data_pipeline.py`: The main script to execute the entire data collection and analysis pipeline.

-   `credibility.py`: Module to flag channels in the database as "credible" based on a master list.

-   `credible_videos.py`: Module to create an intermediate table of videos from credible channels.

-   `labeling.py`: Module to fetch transcripts for credible videos and score them using LLMs.

-   `db_cleanup.py`: A utility script to clean legacy columns from the database tables.

-   `db_utils.py`: A utility script to inspect the database schema and export tables to Excel.

-   `requirements.txt`: A list of all Python dependencies.

-   `/Data/`: Directory containing the master file of credible organizations (`Transplants_Team_Master_File.xlsx`).

-   `/Keywords/`: Directory containing CSV files with search terms (`relevant_keywords.csv`, `reddit_posts_keywords_dedup.csv`).

-   `/Evaluation/`: Directory containing the Python modules for the scoring frameworks (e.g., `HONcodeEvaluate.py`).

Getting Started
---------------

### 1\. Prerequisites

-   Python 3.9+

-   Access to Google Cloud Platform to obtain YouTube Data API v3 and Gemini API keys.

### 2\. Setup

**Clone the repository:**

```
git clone <your-repository-url>
cd <your-repository-name>
```

**Install dependencies:**

It is highly recommended to use a virtual environment.

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

pip install -r requirements.txt
```

**Set up API Keys:**

This pipeline requires two API keys from your Google Cloud project:

1.  **YouTube Data API v3 Key:** For searching videos and channels.

2.  **Gemini API Key:** For running the evaluation models.

For security, store these keys as environment variables.

-   **On macOS/Linux:**

    ```
    export YOUTUBE_API_KEY="your_youtube_api_key_here"
    export GEMINI_API_KEY="your_gemini_api_key_here"
    ```

-   **On Windows (PowerShell):**

    ```
    $Env:YOUTUBE_API_KEY="your_youtube_api_key_here"
    $Env:GEMINI_API_KEY="your_gemini_api_key_here"
    ```

The script is configured to read these environment variables.

### 3\. Running the Pipeline

To run the entire data collection and analysis process from start to finish, simply execute the main pipeline script:

```
python youtube_data_pipeline.py
```

The script will perform the following steps automatically:

1.  Collect video/channel data from YouTube based on your keywords.

2.  Flag channels based on the master credibility file.

3.  Create an intermediate table of credible videos.

4.  Fetch transcripts and generate scores for each credible video, saving the final results to the `CredibleVideosWithScores` table in `youtube_data.db`.

Database Utilities
------------------

Several utility scripts are provided to help manage and inspect the database.

### Inspecting the Database Schema

To see the structure of all tables in the database, run the `db_utils.py` script. This will print the table names and their column schemas.

```
python db_utils.py
```

### Exporting All Tables to Excel

The `db_utils.py` script can also export every table in the database into a single, multi-sheet Excel file named `youtube_database_export.xlsx`.

```
python db_utils.py
```

### Cleaning Up Legacy Columns

If you need to remove old `HONcode_` or `DISCERN_` columns from the `Videos` or `CredibleVideos` tables, run the cleanup script.

```
python db_cleanup.py
```

This will safely recreate the tables without the legacy columns while preserving all other data.