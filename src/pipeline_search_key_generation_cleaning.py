import pandas as pd
import os
from datetime import datetime
import search_key_relevance_prediction
import search_key_clean_deduplicate
import google_autocom_search_key
import google_trends_search_key_generation


def merge_all_csv_files():
    """
    Merge all generated CSV files into one file and delete original files
    """
    raw_dir = "../data/search key/raw"
    all_dataframes = []
    
    # Get all CSV files that need to be merged
    csv_files = []
    
    # Add google_complete_*.csv files
    google_complete_files = [f for f in os.listdir(raw_dir) if f.startswith('google_complete_') and f.endswith('.csv')]
    csv_files.extend(google_complete_files)
    
    # Add google_trends_all_uncleaned.csv file
    trends_file = "google_trends_all_uncleaned.csv"
    if os.path.exists(os.path.join(raw_dir, trends_file)):
        csv_files.append(trends_file)
    
    print(f"Found {len(csv_files)} CSV files to merge:")
    for file in csv_files:
        print(f"  - {file}")
    
    # Read and merge all CSV files
    for file in csv_files:
        filepath = os.path.join(raw_dir, file)
        try:
            df = pd.read_csv(filepath)
            all_dataframes.append(df)
            print(f"✓ Successfully read: {file} ({len(df)} rows)")
        except Exception as e:
            print(f"✗ Failed to read: {file} - {e}")
    
    if all_dataframes:
        # Merge all dataframes
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Remove duplicates
        merged_df = merged_df.drop_duplicates(subset=['search key'], keep='first')
        
        # Save merged file
        output_file = os.path.join(raw_dir, "search_key_combined_unclean.csv")
        merged_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\nMerge completed!")
        print(f"Total records: {len(merged_df)}")
        print(f"Saved to: {output_file}")
        
        # Delete original files
        print(f"\nDeleting original files:")
        for file in csv_files:
            filepath = os.path.join(raw_dir, file)
            try:
                os.remove(filepath)
                print(f"✓ Successfully deleted: {file}")
            except Exception as e:
                print(f"✗ Failed to delete: {file} - {e}")
        
        return output_file
    else:
        print("No CSV files found to merge")
        return None

if __name__ == "__main__":
    #1: get search keys from google autocom and google trends; merge them into one csv file
    for type in ("heart transplant", "kidney transplant", "liver transplant", "pancreas transplant", "heart-lung transplant","lung transplant"):
        results = google_autocom_search_key.get_suggestions_combined(type, max_workers=8, top_n=15)
        for i, suggestion in enumerate(results[:20], 1):
            print(f"{i:2d}. {suggestion}")
        if len(results) > 20:
            print(f"... and {len(results) - 20} more results")
        google_autocom_search_key.save_to_csv(results, type=type)
    
    base_keywords = ["heart transplant", "liver transplant", "kidney transplant","lung transplant", "pancreas transplant"]
    df = google_trends_search_key_generation.get_google_trends_data(base_keywords, data_dir="../data/search key/google_trends")

    df.to_csv("../data/search key/raw/google_trends_all_uncleaned.csv", index=False)
    
    # Merge all generated CSV files
    print("\n" + "="*50)
    print("Starting to merge all CSV files...")
    merge_all_csv_files()
    
    
    # 2: clean and deduplicate the search keys
    df = pd.read_csv('../data/search key/raw/search_key_combined_unclean.csv')
     # load the models
    print("Loading models...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    sbert_model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
    spacy_nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    print("All models loaded.")

    # Step 1: Filter roughly medical terms (with parallel processing)
    print("Using parallel processing for medical term filtering...")
    df_filtered = search_key_clean_deduplicate.filter_medical_terms_parallel(
        df, 
        classifier=classifier, 
        batch_size=16,  # Process 16 words per batch
        max_workers=4   # Use 4 parallel threads
    )
    print("After medical filtering:", len(df_filtered))
    
    # Step 2: insert back the type keywords
    df_inserted = search_key_clean_deduplicate.insert_type_keyword(df_filtered)
    
    # Step 3: Exact Deduplication
    df_cleaned = search_key_clean_deduplicate.exact_deduplication(df_inserted)
    print("After exact deduplication:", len(df_cleaned))
    
    # Step 4: Semantic Deduplication
    df_final = search_key_clean_deduplicate.deduplicate_keywords(
            df=df_cleaned,
            keyword_column='search key',
            model=sbert_model,
            nlp=spacy_nlp,
            distance_threshold=0.15 
        )
    print("After deduplication:", len(df_final))
    df_final.to_csv('../data/search key/raw/search_key_combined_cleaned.csv', index=False)
    print("SUCCESS: Saved to search_key_combined_cleaned.csv")
    
    # 3: predict the relevance of the search keys
    # Define model paths (relative to project root)
    # Get the directory of this script and go up one level to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    model_path = os.path.join(project_root, 'model', 'best_weighted_model_20250926_192011.pkl')
    embedding_model_path = os.path.join(project_root, 'model', 'embedding_model_20250926_192012.pkl')
    
    # Process all search key files
    csv_path = os.path.join(project_root, 'data', 'search key', 'raw', 'search_key_combined_cleaned.csv')
    data_dir = os.path.join(project_root, 'data', 'search key', 'raw')
    search_key_relevance_prediction.process_search_key_files(model_path, embedding_model_path, data_dir=data_dir, csv_path=csv_path)
    
    # Load the prediction results and export relevant keywords
    df = pd.read_csv('../data/search key/raw/predicted_search_key_combined_cleaned.csv')
    
    # Filter and export relevant keywords
    df_relevant = df[df["predicted_relevance"] == 1][["search key", "type"]]
    df_relevant.to_csv('../data/search key/raw/cleaned_search_key.csv', index=False)
    
