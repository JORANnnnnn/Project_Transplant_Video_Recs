# Google Suggest API - Fast search suggestions retrieval
import requests
import string
import concurrent.futures
import pandas as pd
from datetime import datetime

def get_google_suggestions(query, top_n=15):
    """
    Fast retrieval of Google search suggestions using Suggest API
    
    Args:
        query: Search query string
        top_n: Maximum number of suggestions to return
    
    Returns:
        list: List of search suggestions
    """
    url = "http://suggestqueries.google.com/complete/search"
    params = {
        'client': 'chrome',
        'q': query,
        'hl': 'en'
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            # Response format: [query, [suggestions], ...]
            data = response.json()
            suggestions = data[1] if len(data) > 1 else []
            # Return only top_n results
            return suggestions[:top_n]
    except:
        pass
    
    return []

def get_suggestions_combined(base_query, *, max_workers=10, top_n=10):
    """
    Combined strategy for fetching search suggestions with deduplication and sorting
    
    Args:
        base_query: Base search query
        max_workers: Number of concurrent workers
        top_n: Maximum suggestions per query variant
    
    Returns:
        list: Cleaned and sorted suggestions
    """
    
    # 1) Search base query itself 
    base_queries = [base_query]
    
    # 2) Search base query + a-z extensions
    letters = list(string.ascii_lowercase)
    az_queries = [f"{base_query} {ch}" for ch in letters]
   
    # 3) Search base query + question words
    question_words = ['what', 'how', 'when', 'where', 'why', 'who']
    question_queries = [f"{qw} {base_query}" for qw in question_words]
    
    
    
    # 4) Combine all queries
    all_queries = list({*base_queries, *az_queries, *question_queries})
    
    print(f"Generated {len(all_queries)} query variants, max {top_n} suggestions per query")
    
    # 5) Concurrently fetch all suggestions
    all_results = set()
    
    def fetch_suggestions(query):
        try:
            return get_google_suggestions(query, top_n)
        except Exception:
            return []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_suggestions, q): q for q in all_queries}
        
        for future in concurrent.futures.as_completed(futures):
            query = futures[future]
            try:
                suggestions = future.result()
                all_results.update(suggestions)
                print(f"✓ Completed: {query} ({len(suggestions)} results)")
            except Exception as exc:
                print(f"✗ Failed: {query} - {exc}")
    
    # 6) Deduplicate, filter whitespace, and sort
    clean_results = sorted({s.strip() for s in all_results if isinstance(s, str) and s.strip()})
    
    return clean_results

def save_to_csv(cluster_results, type=None):
    """
    Save cluster results to CSV file
    
    Args:
        cluster_results: List of cluster results
        type: Type of search key
    
    Returns:
        str: Saved filename
    """
    import os
    
    # Create data/search key/raw directory if it doesn't exist
    output_dir = "../data/search key/raw"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"google_complete_{type}.csv"
    filepath = os.path.join(output_dir, filename)

    # Convert to DataFrame with proper column name
    df = pd.DataFrame({"search key": cluster_results})
    df["type"] = type
    df["source"] = "google_complete"
    
    # Save to CSV
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"Data saved to: {filepath}")
    return filepath




# Test function
if __name__ == "__main__":
    # Get top 10 suggestions per query
    for type in ("heart transplant", "kidney transplant", "liver transplant", "pancreas transplant", "heart-lung transplant","lung transplant"):
        results = get_suggestions_combined(type, max_workers=8, top_n=15)
        for i, suggestion in enumerate(results[:20], 1):
            print(f"{i:2d}. {suggestion}")
        if len(results) > 20:
            print(f"... and {len(results) - 20} more results")
        save_to_csv(results, type=type)
    