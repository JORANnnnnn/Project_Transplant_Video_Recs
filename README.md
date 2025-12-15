# Transplant Video Education Browser

A Streamlit web application for browsing and searching patient education videos about organ transplantation. This application provides a curated collection of high-quality, LLM-recommended transplant education videos with advanced filtering, sorting, and ranking capabilities.

## Features

### 🔍 Search & Discovery
- **Full-text search** across video titles, descriptions, channels, and captions
- **Smart filtering** by organization type, location, and video length
- **Category-based organization** with collapsible sections

### 📊 Ranking & Metrics
- **LLM Ranking**: Videos ranked by AI recommendation scores
- **PEMAT Score**: Patient Education Materials Assessment Tool scores
- **Final Score**: Composite ranking based on multiple factors
- **Engagement Metrics**: Views, likes, comments, and engagement scores
- **Understandability**: Videos marked as understandable for patients

### 🎯 Filtering Options
- **Organization Type**:
  - Only Transplant Centers
  - Only USA Organizations
  - Only Non-Profits
- **Video Length**: Short (<4m), Medium (4–20m), Long (>20m)
- **Sorting**: By ranking, engagement, views, likes, comments, duration, or alphabetically

### 📹 Video Features
- **Embedded video player** - Watch videos directly in the app
- **Transcript viewing** - Fetch and view video transcripts
- **Detailed metrics** - Views, likes, comments, duration, engagement scores
- **Category information** - Patient education stages (consideration, evaluation, waitlist, surgery, post-surgery)

## Dataset

The application uses the `youtube_recommendations_topK_v3_merged.xlsx` dataset, which contains:
- **300 top-ranked recommended videos** from LLM analysis
- **Patient education videos** across all transplant stages
- **Ranking data**: LLM Ranking, PEMAT Score, final_score, category_rank
- **Video metadata**: Titles, descriptions, engagement metrics, channel information

### Categories
- Patient education - consideration
- Patient education - evaluation
- Patient education - surgery
- Patient Education - waitlist
- Patient education - post surgery

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/JORANnnnnn/Project_Transplant_Video_Recs.git
cd Project_Transplant_Video_Recs
git checkout website
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure the dataset file is present:
   - `youtube_recommendations_topK_v3_merged.xlsx` should be in the project root

5. Run the application:
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Requirements

See `requirements.txt` for the full list of dependencies:
- `pandas>=2.3.3` - Data manipulation
- `openpyxl>=3.1.5` - Excel file reading
- `streamlit>=1.52.1` - Web framework
- `requests>=2.32.5` - HTTP requests
- `youtube-transcript-api>=1.2.3` - Video transcript fetching

## Usage

1. **Search**: Use the search bar to find videos by keywords, titles, or descriptions
2. **Filter**: Apply filters for organization type, location, and video length
3. **Sort**: Choose how to sort videos (by ranking, engagement, views, etc.)
4. **Browse**: Explore videos organized by category and transplant center
5. **Watch**: Click on videos to watch them embedded in the app
6. **Transcripts**: Click "View Transcript" to fetch and read video transcripts

## Deployment

This application can be deployed on:
- **Streamlit Cloud**: Connect your GitHub repository and deploy automatically
- **Heroku**: Use the Procfile and requirements.txt
- **Other platforms**: Any platform that supports Python and Streamlit

### Streamlit Cloud Deployment

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Select the `website` branch
5. Set the main file to `app.py`
6. Deploy!

## Data Source

The dataset is generated from:
- YouTube video metadata
- LLM-based categorization and recommendation
- Patient education material assessment (PEMAT) scoring
- Engagement metrics analysis

## Notes

- The app filters out promotional videos automatically
- Only LLM-recommended videos are shown
- Transcript fetching requires the `youtube-transcript-api` package
- Some videos may not have transcripts available due to YouTube API limitations

## License

[Add your license here]

## Contact

[Add contact information here]
