import re
import math
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from collections import defaultdict

import pandas as pd
import streamlit as st

# --------------------------- Page config ---------------------------
st.set_page_config(page_title="Transplant Video Browser", layout="wide", initial_sidebar_state="collapsed")
st.title("🔍 Search Transplant Education Videos")

# --------------------------- Caching -------------------------------
@st.cache_data(show_spinner=False)
def load_recommended_data(file_path: Path) -> pd.DataFrame:
    """Load Excel with top K recommendations (already filtered)."""
    if not file_path.exists():
        st.error(f"File not found: {file_path}")
        st.stop()
    xls = pd.ExcelFile(file_path)
    sheet = xls.sheet_names[0]
    df = pd.read_excel(file_path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    
    # The merged topK dataset already contains only recommended videos
    # No need to filter further
    
    return df.reset_index(drop=True)

# --------------------------- Helper Functions -------------------------------
def parse_iso8601_duration(s: Optional[str]) -> Optional[int]:
    """Parse ISO8601 duration (PT3M35S) to seconds."""
    if not isinstance(s, str) or not s.startswith('PT'):
        return None
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s)
    if not m:
        return None
    h, m_, s_ = (int(m.group(1) or 0), int(m.group(2) or 0), int(m.group(3) or 0))
    return h*3600 + m_*60 + s_

def format_duration(seconds: Optional[int]) -> str:
    """Format seconds to HH:MM:SS or MM:SS."""
    if seconds is None:
        return "—"
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"

def duration_bucket(seconds: Optional[int]) -> Optional[str]:
    """Categorize video length."""
    if seconds is None:
        return None
    if seconds < 4*60:
        return "Short (<4m)"
    if seconds <= 20*60:
        return "Medium (4–20m)"
    return "Long (>20m)"

def safe_int(x) -> Optional[int]:
    """Safely convert to int."""
    try:
        return int(float(x)) if pd.notna(x) else None
    except (ValueError, TypeError):
        return None

def compute_engagement_score(row: pd.Series) -> float:
    """Compute engagement score: log(views+1) + 2*log(likes+1) + 3*log(comments+1)."""
    v = safe_int(row.get('view_count')) or 0
    l = safe_int(row.get('like_count')) or 0
    c = safe_int(row.get('comment_count')) or 0
    score = math.log1p(v) + 2.0*math.log1p(l) + 3.0*math.log1p(c)
    return round(score, 4)

def is_usa_organization(channel_title: str) -> bool:
    """Heuristic to detect USA organizations."""
    if pd.isna(channel_title):
        return False
    channel_lower = str(channel_title).lower()
    usa_indicators = ['mayo', 'cleveland', 'johns hopkins', 'mass general', 'stanford', 
                     'uc health', 'upmc', 'duke', 'ucla', 'michigan medicine', 'ohio state',
                     'penn medicine', 'nyp', 'newyork-presbyterian', 'henry ford', 'georgetown',
                     'froedtert', 'methodist', 'u of u health', 'wexner', 'cedars-sinai',
                     'loyola', 'uva', 'american', 'national kidney foundation']
    return any(indicator in channel_lower for indicator in usa_indicators)

def is_transplant_center(channel_title: str) -> bool:
    """Heuristic to detect transplant centers."""
    if pd.isna(channel_title):
        return False
    channel_lower = str(channel_title).lower()
    transplant_indicators = ['transplant', 'mayo', 'cleveland', 'johns hopkins', 'mass general',
                            'stanford', 'uc health', 'upmc', 'duke', 'ucla', 'michigan medicine',
                            'ohio state', 'penn medicine', 'nyp', 'newyork-presbyterian',
                            'henry ford', 'georgetown', 'froedtert', 'methodist', 'u of u health',
                            'wexner', 'cedars-sinai', 'loyola', 'uva']
    return any(indicator in channel_lower for indicator in transplant_indicators)

def is_nonprofit(channel_title: str) -> bool:
    """Heuristic to detect non-profit organizations."""
    if pd.isna(channel_title):
        return False
    channel_lower = str(channel_title).lower()
    nonprofit_indicators = ['foundation', 'national kidney foundation', 'american liver foundation',
                           'mayo', 'cleveland', 'johns hopkins', 'mass general', 'stanford']
    return any(indicator in channel_lower for indicator in nonprofit_indicators)

@st.cache_data(show_spinner=False)
def try_fetch_transcript(video_id: str, languages: Tuple[str, ...] = ("en", "en-US", "en-GB")) -> Tuple[bool, str]:
    """Fetch transcript for a video."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
    except Exception:
        return (False, "Transcript module not available.")

    try:
        langs = list(languages)
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            data = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
        else:
            api = YouTubeTranscriptApi()
            fetched = api.fetch(video_id, languages=langs)
            if hasattr(fetched, "to_dict"):
                data = fetched.to_dict()
            elif hasattr(fetched, "to_raw_data"):
                data = fetched.to_raw_data()
            else:
                data = list(fetched)

        text = " ".join(d.get("text", "") for d in data if isinstance(d, dict) and d.get("text")).strip()
        return (True, text if text else "Transcript is empty.")
    except (NoTranscriptFound, TranscriptsDisabled):
        return (False, "Transcript unavailable for this video.")
    except Exception as e:
        error_str = str(e).lower()
        error_msg = str(e)
        
        # Check for IP blocking or request blocking errors
        if any(keyword in error_str for keyword in ["ip", "blocked", "requestblocked", "could not retrieve", "youtube is blocking"]):
            return (False, 
                    "⚠️ **YouTube IP Blocking Detected**\n\n"
                    "YouTube is blocking transcript requests from this IP address. This usually happens due to:\n\n"
                    "• Too many requests from this IP\n"
                    "• Using a cloud provider IP (AWS, GCP, Azure, etc.)\n\n"
                    "**Possible Solutions:**\n"
                    "• Wait a few minutes and try again\n"
                    "• Use a different network or VPN\n"
                    "• Check if the video has captions available directly on YouTube\n"
                    "• Consider using proxies (see youtube-transcript-api documentation)")
        
        # Check for too many requests
        if "too many" in error_str or "rate limit" in error_str:
            return (False, "⚠️ Too many requests. Please wait a moment and try again later.")
        
        # Generic error
        return (False, f"⚠️ Transcript error: {error_msg}")

def filter_contains_any(df: pd.DataFrame, cols: List[str], q: str) -> pd.DataFrame:
    """Case-insensitive substring match across columns."""
    if not q.strip():
        return df
    ql = q.strip().lower()
    mask = None
    for c in cols:
        if c in df.columns:
            col_vals = df[c].astype(str).str.lower()
            m = col_vals.str.contains(ql, na=False)
            mask = m if mask is None else (mask | m)
    return df[mask] if mask is not None else df

# --------------------------- Load Data -------------------------------
data_path = Path("youtube_videos_merged_topK.xlsx")
df = load_recommended_data(data_path)

# Process data
df['duration_seconds'] = df['duration'].apply(parse_iso8601_duration)
df['duration_formatted'] = df['duration_seconds'].apply(format_duration)
df['duration_bucket'] = df['duration_seconds'].apply(duration_bucket)
df['engagement_score'] = df.apply(compute_engagement_score, axis=1)
df['is_usa'] = df['channel_title'].apply(is_usa_organization)
df['is_transplant_center'] = df['channel_title'].apply(is_transplant_center)
df['is_nonprofit'] = df['channel_title'].apply(is_nonprofit)
df['video_url'] = df['video_id'].apply(lambda vid: f"https://www.youtube.com/watch?v={vid}" if pd.notna(vid) else "")

# Check transcript availability
try:
    import youtube_transcript_api  # noqa: F401
    HAVE_TRANSCRIPT_API = True
except Exception:
    HAVE_TRANSCRIPT_API = False

# --------------------------- Video Card Renderer -------------------------------
def render_video_card(row: pd.Series, idx: int):
    """Render a single video card with embed, metrics, and transcript."""
    video_id = row.get('video_id', '')
    title = row.get('title', 'Untitled Video')
    channel = row.get('channel_title', 'Unknown Channel')
    description = row.get('description', '')
    views = safe_int(row.get('view_count'))
    likes = safe_int(row.get('like_count'))
    comments = safe_int(row.get('comment_count'))
    duration = row.get('duration_formatted', '—')
    engagement = row.get('engagement_score', 0)
    video_url = row.get('video_url', '')
    # Handle merged category columns
    category = (row.get('Category_gpt_5_topk') or 
                row.get('Category_gpt_5_gpt5') or 
                row.get('Category_gpt_5') or 
                row.get('Category') or 
                'Unknown')
    
    st.markdown("---")
    st.markdown(f"#### {title}")
    
    # Video embed
    if video_url:
        st.video(video_url)
    
    # Get ranking metrics
    llm_ranking = row.get('LLM Ranking')
    final_score = row.get('final_score')
    pemat_score = row.get('PEMAT Score')
    is_understandable = row.get('is_understandable')
    
    # Metrics in columns (expanded to show ranking)
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5, metric_col6 = st.columns(6)
    
    with metric_col1:
        st.metric("👁️ Views", f"{views:,}" if views else "—")
    with metric_col2:
        st.metric("👍 Likes", f"{likes:,}" if likes else "—")
    with metric_col3:
        st.metric("💬 Comments", f"{comments:,}" if comments else "—")
    with metric_col4:
        st.metric("⏱️ Length", duration)
    with metric_col5:
        if llm_ranking is not None and pd.notna(llm_ranking):
            st.metric("🏆 LLM Rank", f"#{int(llm_ranking)}")
        elif final_score is not None and pd.notna(final_score):
            st.metric("⭐ Final Score", f"{final_score:.3f}")
        else:
            st.metric("⭐ Engagement", f"{engagement:.2f}")
    with metric_col6:
        if pemat_score is not None and pd.notna(pemat_score):
            st.metric("📊 PEMAT", f"{pemat_score:.2f}")
        else:
            st.metric("⭐ Engagement", f"{engagement:.2f}")
    
    # Channel, category, and ranking info
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.caption(f"📺 Channel: {channel}")
    with info_col2:
        st.caption(f"📂 Category: {category}")
    with info_col3:
        if is_understandable is not None and pd.notna(is_understandable):
            understand_text = "✅ Understandable" if is_understandable else "❌ Not Understandable"
            st.caption(understand_text)
        elif final_score is not None and pd.notna(final_score):
            st.caption(f"⭐ Score: {final_score:.3f}")
    
    # Description
    if description and str(description).strip() and str(description) != 'nan':
        with st.expander("📝 Description", expanded=False):
            st.write(str(description)[:500] + ("..." if len(str(description)) > 500 else ""))
    
    # Transcript
    if HAVE_TRANSCRIPT_API and video_id:
        transcript_key = f"transcript_{video_id}_{idx}"
        if transcript_key not in st.session_state:
            st.session_state[transcript_key] = None
        
        with st.expander("📄 View Transcript", expanded=False):
            if st.button("Fetch Transcript", key=f"fetch_tx_{video_id}_{idx}"):
                with st.spinner("Fetching transcript..."):
                    ok, text = try_fetch_transcript(video_id)
                    if ok:
                        st.session_state[transcript_key] = text
                        st.text_area("Transcript", value=text, height=300, key=f"tx_area_{video_id}_{idx}")
                        st.download_button(
                            "Download Transcript",
                            data=text,
                            file_name=f"{video_id}_transcript.txt",
                            key=f"dl_{video_id}_{idx}"
                        )
                    else:
                        # Display error message (may contain markdown formatting)
                        if "⚠️" in text or "**" in text:
                            st.markdown(text)
                        else:
                            st.error(text)
            elif st.session_state[transcript_key]:
                st.text_area("Transcript", value=st.session_state[transcript_key], height=300, key=f"tx_area_{video_id}_{idx}")
                st.download_button(
                    "Download Transcript",
                    data=st.session_state[transcript_key],
                    file_name=f"{video_id}_transcript.txt",
                    key=f"dl_{video_id}_{idx}"
                )
    elif video_id:
        with st.expander("📄 View Transcript", expanded=False):
            st.info("Transcript feature requires 'youtube-transcript-api' package.")
    
    # External link
    if video_url:
        st.markdown(f"[🔗 Watch on YouTube]({video_url})")
    
    st.markdown("")

# --------------------------- Main Interface: Search & Filters (Horizontal) -------------------------------
st.markdown("---")

# Search bar
search_query = st.text_input(
    "🔍 Search anything and get results from it",
    placeholder="Search by title, description, channel, or keywords...",
    key="main_search"
)

# Filters in columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("📋 Organization Type")
    org_transplant_only = st.checkbox("Only Transplant Centers", value=False, key="org_transplant")
    org_usa_only = st.checkbox("Only USA Organizations", value=False, key="org_usa")
    org_nonprofit_only = st.checkbox("Only Non-Profits", value=False, key="org_nonprofit")

with col2:
    st.subheader("⏱️ Video Length")
    length_options = ["Short (<4m)", "Medium (4–20m)", "Long (>20m)"]
    length_filter = st.multiselect(
        "Choose video length",
        length_options,
        default=[],
        key="length_filter"
    )

with col3:
    st.subheader("📊 Sort By")
    sort_options = [
        "LLM Ranking (Best First)",
        "Final Score (Highest First)",
        "PEMAT Score (Highest First)",
        "Engagement Score (Most Reputable)",
        "Views (Most Watched)",
        "Likes (Most Liked)",
        "Comments (Most Discussed)",
        "Duration (Longest First)",
        "Duration (Shortest First)",
        "Channel Name (A→Z)",
        "Title (A→Z)"
    ]
    sort_choice = st.selectbox("Sort videos by", sort_options, key="sort_choice")

with col4:
    st.subheader("🎯 Results")
    max_videos = st.slider("Number of videos to show", min_value=10, max_value=15, value=12, step=1, key="max_videos")

st.markdown("---")

# --------------------------- Apply Filters -------------------------------
work = df.copy()

# Filter out promotional videos
promo_keywords = ['Promotional', 'promotional', 'promo']
category_cols = [c for c in work.columns if 'category' in c.lower() and 'Category' in c]
for cat_col in category_cols:
    if cat_col in work.columns:
        work = work[~work[cat_col].astype(str).str.contains('promo', case=False, na=False)]
        break  # Use first matching category column

# Search filter
if search_query:
    searchable_cols = ['title', 'description', 'channel_title', 'search_term', 'caption_text']
    work = filter_contains_any(work, searchable_cols, search_query)

# Organization filters
if org_transplant_only:
    work = work[work['is_transplant_center'] == True]
if org_usa_only:
    work = work[work['is_usa'] == True]
if org_nonprofit_only:
    work = work[work['is_nonprofit'] == True]

# Length filter
if length_filter:
    work = work[work['duration_bucket'].isin(length_filter)]

# Sorting
if sort_choice == "LLM Ranking (Best First)":
    work = work.sort_values('LLM Ranking', ascending=True, na_position='last')  # Lower rank number = better
elif sort_choice == "Final Score (Highest First)":
    work = work.sort_values('final_score', ascending=False, na_position='last')
elif sort_choice == "PEMAT Score (Highest First)":
    work = work.sort_values('PEMAT Score', ascending=False, na_position='last')
elif sort_choice == "Engagement Score (Most Reputable)":
    work = work.sort_values('engagement_score', ascending=False)
elif sort_choice == "Views (Most Watched)":
    work['_views'] = work['view_count'].apply(safe_int)
    work = work.sort_values('_views', ascending=False).drop(columns=['_views'])
elif sort_choice == "Likes (Most Liked)":
    work['_likes'] = work['like_count'].apply(safe_int)
    work = work.sort_values('_likes', ascending=False).drop(columns=['_likes'])
elif sort_choice == "Comments (Most Discussed)":
    work['_comments'] = work['comment_count'].apply(safe_int)
    work = work.sort_values('_comments', ascending=False).drop(columns=['_comments'])
elif sort_choice == "Duration (Longest First)":
    work = work.sort_values('duration_seconds', ascending=False, na_position='last')
elif sort_choice == "Duration (Shortest First)":
    work = work.sort_values('duration_seconds', ascending=True, na_position='last')
elif sort_choice == "Channel Name (A→Z)":
    work = work.sort_values('channel_title', na_position='last')
elif sort_choice == "Title (A→Z)":
    work = work.sort_values('title', na_position='last')

# Limit results
work = work.head(max_videos).reset_index(drop=True)

# --------------------------- Group by Category (LLM Categories) -------------------------------
# Check for category column (could be from topK or gpt5 merge)
if 'Category_gpt_5_topk' in work.columns:
    category_col = 'Category_gpt_5_topk'
elif 'Category_gpt_5_gpt5' in work.columns:
    category_col = 'Category_gpt_5_gpt5'
elif 'Category_gpt_5' in work.columns:
    category_col = 'Category_gpt_5'
elif 'Category' in work.columns:
    category_col = 'Category'
else:
    category_col = None

if category_col and not work.empty:
    # Filter out invalid categories
    valid_categories = work[~work[category_col].isin(['Error', 'INVALID_CATEGORY', None])]
    
    if not valid_categories.empty:
        # Group by category, then by channel (transplant center/company)
        categories = valid_categories[category_col].unique()
        
        # Limit to 4-5 main categories
        main_categories = [
            'Patient education - post surgery',
            'Patient education - consideration',
            'Patient education - evaluation',
            'Patient Education - waitlist',
            'Patient education - surgery'
        ]
        
        # Show main patient education categories first, then others
        categories_to_show = [c for c in main_categories if c in categories][:5]
        other_categories = [c for c in categories if c not in categories_to_show]
        categories_to_show.extend(other_categories[:5-len(categories_to_show)])
        
        st.markdown(f"### 📚 Results: {len(work)} videos found")
        
        for category in categories_to_show:
            cat_videos = valid_categories[valid_categories[category_col] == category]
            if cat_videos.empty:
                continue
            
            # Group by channel (transplant center/company)
            channels = cat_videos['channel_title'].unique()
            
            with st.expander(f"📁 {category} ({len(cat_videos)} videos)", expanded=False):
                for channel in sorted(channels):
                    channel_videos = cat_videos[cat_videos['channel_title'] == channel]
                    
                    with st.expander(f"🏥 {channel} ({len(channel_videos)} videos)", expanded=False):
                        for idx, row in channel_videos.iterrows():
                            render_video_card(row, idx)
    else:
        st.info("No videos found matching your criteria.")
        work = pd.DataFrame()
else:
    # Fallback: group by channel if no category column
    if not work.empty:
        st.markdown(f"### 📚 Results: {len(work)} videos found")
        channels = work['channel_title'].unique()
        
        for channel in sorted(channels):
            channel_videos = work[work['channel_title'] == channel]
            
            with st.expander(f"🏥 {channel} ({len(channel_videos)} videos)", expanded=False):
                for idx, row in channel_videos.iterrows():
                    render_video_card(row, idx)
    else:
        st.info("No videos found matching your criteria.")

# Footer
if not work.empty:
    st.markdown("---")
    st.caption("💡 Tip: Use the search bar and filters above to find specific transplant education videos. Videos are sorted by engagement metrics to show the most reputable content first.")
