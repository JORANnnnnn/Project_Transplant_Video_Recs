# Use the pkl model to predict the relevance of the search keys
# 1 as relevant, 0 as irrelevant
# save the predicted results to a csv file
# the file name would be the current file name + _predicted.csv
# the csv file would have four columns: search key, predicted relevance, type, source

import pandas as pd
import joblib
import os
import numpy as np

# Try to import sentence-transformers for fallback embedding
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("sentence-transformers not available, using saved embedding model only")

def load_model(model_path):
    # load the pkl model
    model = joblib.load(model_path)
    return model

def load_csv(csv_path):
    # load the csv file
    df = pd.read_csv(csv_path)
    return df

def load_embedding_model(embedding_model_path):
    # load the embedding model
    embedding_model = joblib.load(embedding_model_path)
    return embedding_model

def get_embeddings(embedding_model, search_keys, fallback_model_name='all-MiniLM-L6-v2'):
    """
    Convert search keys to embeddings using the embedding model
    
    Args:
        embedding_model: The loaded embedding model
        search_keys: List of search key strings
        fallback_model_name: Fallback model name for sentence-transformers
    
    Returns:
        numpy array: Embeddings for the search keys
    """
    try:
        # Try the loaded embedding model first
        if hasattr(embedding_model, 'encode'):
            embeddings = embedding_model.encode(search_keys)
            print(f"Successfully generated embeddings with shape: {embeddings.shape}")
            return embeddings
        elif hasattr(embedding_model, 'transform'):
            embeddings = embedding_model.transform(search_keys)
            print(f"Successfully generated embeddings with shape: {embeddings.shape}")
            return embeddings
        else:
            print("Embedding model doesn't have encode or transform method")
            raise AttributeError("No suitable method found")
            
    except Exception as e:
        print(f"Error with saved embedding model: {e}")
        print(f"Embedding model type: {type(embedding_model)}")
        
        # Try fallback with sentence-transformers
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print(f"Trying fallback with sentence-transformers model: {fallback_model_name}")
                fallback_model = SentenceTransformer(fallback_model_name)
                embeddings = fallback_model.encode(search_keys)
                print(f"Successfully generated embeddings with fallback model, shape: {embeddings.shape}")
                return embeddings
            except Exception as fallback_error:
                print(f"Fallback embedding also failed: {fallback_error}")
                return None
        else:
            print("No fallback available")
            return None

def predict_relevance(model, embeddings):
    """
    Predict the relevance of search keys using their embeddings
    
    Args:
        model: The trained classification model
        embeddings: Embeddings of the search keys
    
    Returns:
        numpy array: Predicted relevance scores (0 or 1)
    """
    try:
        predicted_relevance = model.predict(embeddings)
        return predicted_relevance
    except Exception as e:
        print(f"Error predicting relevance: {e}")
        return None

def save_predicted_results(original_name, predicted_relevance, df, output_dir='../data/search key'):
    """
    Save the predicted results to a csv file
    
    Args:
        original_name: Original filename
        predicted_relevance: Predicted relevance scores
        df: DataFrame with search keys
        output_dir: Output directory for predicted files
    
    Returns:
        str: Path to saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add predicted relevance to dataframe
    df['predicted_relevance'] = predicted_relevance
    
    # Create output filename
    base_name = original_name.replace('.csv', '')
    output_filename = f'predicted_{base_name}.csv'
    output_path = os.path.join(output_dir, output_filename)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Predicted results saved to: {output_path}")
    return output_path

def process_search_key_files(model_path, embedding_model_path, data_dir='../data/search key'):
    """
    Process all CSV files in the data directory and predict relevance
    
    Args:
        model_path: Path to the trained model
        embedding_model_path: Path to the embedding model
        data_dir: Directory containing CSV files
    """
    # Load models
    print("Loading models...")
    model = load_model(model_path)
    embedding_model = load_embedding_model(embedding_model_path)
    
    # Process each CSV file
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and not f.startswith('predicted_')]
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file}")
        
        try:
            # Load CSV file
            csv_path = os.path.join(data_dir, csv_file)
            df = load_csv(csv_path)
            
            # Get search keys
            search_keys = df['search key'].tolist()
            print(f"Processing {len(search_keys)} search keys...")
            
            # Get embeddings
            embeddings = get_embeddings(embedding_model, search_keys)
            if embeddings is None:
                print(f"Failed to get embeddings for {csv_file}")
                continue
            
            # Predict relevance
            predicted_relevance = predict_relevance(model, embeddings)
            if predicted_relevance is None:
                print(f"Failed to predict relevance for {csv_file}")
                continue
            
            # Save results
            save_predicted_results(csv_file, predicted_relevance, df, data_dir)
            
            # Print summary
            relevant_count = np.sum(predicted_relevance)
            total_count = len(predicted_relevance)
            print(f"Summary: {relevant_count}/{total_count} search keys predicted as relevant")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

if __name__ == '__main__':
    # Define model paths
    model_path = '../model/best_weighted_model_20250926_192011.pkl'
    embedding_model_path = '../model/embedding_model_20250926_192012.pkl'
    
    # Process all search key files
    process_search_key_files(model_path, embedding_model_path)
            
