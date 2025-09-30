import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial

def _classify_batch(terms_batch, classifier, candidate_labels):
    """
    对一批术语进行分类的辅助函数
    """
    try:
        results = classifier(terms_batch, candidate_labels, multi_label=False)
        return results
    except Exception as e:
        print(f"Error in batch classification: {e}")
        return None

def filter_medical_terms_parallel(df, classifier=None, candidate_labels=None, batch_size=16, max_workers=4):
    """
    使用并行处理过滤医疗术语的优化版本
    """
    if df.empty:
        return df
    df = df.copy()
    if candidate_labels is None:
        candidate_labels = ["medical", "health", "disease", "entertainment", "celebrity", "politics", "technology"]

    if classifier is None:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    df['search key'] = df['search key'].astype(str).str.lower().str.strip()
    df = df[df['search key'] != ''] 
    if df.empty:
        return df

    terms_to_classify = df['search key'].tolist()
    print(f"Parallel categorizing {len(terms_to_classify)} search keys with {max_workers} workers...")
    
    # 将术语分批
    batches = [terms_to_classify[i:i + batch_size] for i in range(0, len(terms_to_classify), batch_size)]
    print(f"Split into {len(batches)} batches of size {batch_size}")
    
    all_results = []
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有批次任务
        future_to_batch = {
            executor.submit(_classify_batch, batch, classifier, candidate_labels): i 
            for i, batch in enumerate(batches)
        }
        
        # 收集结果
        completed_batches = 0
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_results = future.result()
                if batch_results is not None:
                    all_results.extend(batch_results)
                    completed_batches += 1
                    print(f"Completed batch {batch_idx + 1}/{len(batches)} ({completed_batches}/{len(batches)})")
                else:
                    print(f"Batch {batch_idx + 1} failed")
            except Exception as e:
                print(f"Batch {batch_idx + 1} generated an exception: {e}")
    
    print("Parallel categorizing completed.")
    
    if len(all_results) != len(terms_to_classify):
        print(f"Warning: Expected {len(terms_to_classify)} results, got {len(all_results)}")
        # 如果结果数量不匹配，回退到原始方法
        print("Falling back to sequential processing...")
        return filter_medical_terms(df, classifier, candidate_labels)
    
    df['assigned_label'] = [result['labels'][0] for result in all_results]
    df['assigned_label'] = df['assigned_label'].str.strip().str.lower()
    medical_labels = {"medical", "health", "disease", "technology"}
    df_filtered = df[df['assigned_label'].isin(medical_labels)].copy()

    return df_filtered.drop(columns=['assigned_label']).reset_index(drop=True)

def filter_medical_terms(df, classifier=None, candidate_labels=None):
    """
    Filter terms using zero-shot classification, keeping only medical-related ones.
    """
    if df.empty:
        return df
    df = df.copy()
    if candidate_labels is None:
        candidate_labels = ["medical", "health", "disease", "entertainment", "celebrity", "politics", "technology"]

    if classifier is None:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    df['search key'] = df['search key'].astype(str).str.lower().str.strip()
    df = df[df['search key'] != ''] 
    if df.empty:
        return df

    terms_to_classify = df['search key'].tolist()
    print(f"Categorizing {len(terms_to_classify)} search keys...")
    results = classifier(terms_to_classify, candidate_labels, multi_label=False)
    print("Categorizing completed.")
    df['assigned_label'] = [result['labels'][0] for result in results]
    df['assigned_label'] = df['assigned_label'].str.strip().str.lower()
    medical_labels = {"medical", "health", "disease","technology"}
    df_filtered = df[df['assigned_label'].isin(medical_labels)].copy()

    return df_filtered.drop(columns=['assigned_label']).reset_index(drop=True)
            
# 2. Insert back specific types
def insert_type_keyword(df):
    """
    Combines 'type' keywords into 'search key' for more specific terms.

    It skips the combination if any word from the 'type' string already exists
    in the 'search key' string (case-insensitive).

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with at least 'search key' and 'type' columns.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the 'search key' column modified.
    """
    def _process_row(row):
        # Extract the search key and type, ensuring they are strings
        search_key = str(row['search key'])
        type_keyword = str(row['type'])

        # Normalize both strings to lowercase for case-insensitive comparison
        search_key_lower = search_key.lower()
        type_keyword_lower = type_keyword.lower()

        # Split the type keyword into individual words.
        # e.g., "heart transplant" -> ["heart", "transplant"]
        type_words = type_keyword_lower.split()

        # Check if ANY of the words from the type_keyword exist in the search_key
        # This is the core logic for the skip condition.
        if any(word in search_key_lower for word in type_words):
            # If a match is found, just return the original search key
            return search_key
        else:
            # Otherwise, combine them into the new, more specific search key
            return f"{type_keyword} {search_key}"

    # Apply the processing function to each row of the DataFrame.
    # The result will be a new Series, which we use to update the 'search key' column.
    df['search key'] = df.apply(_process_row, axis=1)
    
    return df
    
# 3. Exact Deduplication
def exact_deduplication(df):
    """
    Deduplicate search terms exactly, merging base_keywords into a list.
    """
    if df.empty:
        return df

    # 使用groupby对'search key'进行分组，然后对其他列进行聚合
    df_cleaned = df.groupby('search key', as_index=False).agg({
        # 对于'type'列，将每个组内的值去重后合并成一个列表
        'type': lambda x: list(x.unique()),
        # 对于'source'列，取每个组的第一个值（假设同一组的source应该相同）
        'source': 'first'
    })
    
    return df_cleaned


# 4. Deduplication by semantic similarity
def deduplicate_keywords(df: pd.DataFrame, 
                                          keyword_column: str,
                                          model: SentenceTransformer, 
                                          nlp, 
                                          distance_threshold: float = 0.2) -> pd.DataFrame:
    """
    对DataFrame中的关键词列进行语义去重,并保留簇内的其他变体。

    此函数返回一个去重后的DataFrame,其中包含每个簇的代表行、
    一个包含该簇所有其他成员的'variants'列，以及'cluster_id'。

    Parameters
    ----------
    df : pd.DataFrame
        输入的DataFrame。
    keyword_column : str
        包含关键词的列的名称。
    model : SentenceTransformer
        预加载的sentence-transformer模型。
    nlp : spacy.Language
        预加载的spacy模型。
    distance_threshold : float, optional
        聚类的距离阈值，默认为0.2。

    Returns
    -------
    pd.DataFrame
        一个去重后的DataFrame，包含原始数据的子集和新的'cluster_id'、'variants'列。
    """
    if df.empty or keyword_column not in df.columns:
        return pd.DataFrame()

    df_processed = df.copy()
    
    # 1. 标准化
    def normalize_text(text):
        doc = nlp(str(text).lower())
        return " ".join([token.lemma_ for token in doc if not token.is_punct])

    df_processed['normalized_keyword'] = df_processed[keyword_column].apply(normalize_text)

    # 2. 向量化 (Embedding)
    unique_keywords = df_processed['normalized_keyword'].unique().tolist()
    embeddings = model.encode(unique_keywords, show_progress_bar=False)
    embedding_map = {kw: emb for kw, emb in zip(unique_keywords, embeddings)}
    df_processed['embedding'] = df_processed['normalized_keyword'].map(embedding_map)

    # 3. 聚类
    if len(unique_keywords) <= 1:
        df_processed['cluster_id'] = 0
    else:
        unique_embeddings = np.array([embedding_map[kw] for kw in unique_keywords])
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='cosine',
            linkage='average'
        ).fit(unique_embeddings)
        
        cluster_map = {kw: cluster_id for kw, cluster_id in zip(unique_keywords, clustering.labels_)}
        df_processed['cluster_id'] = df_processed['normalized_keyword'].map(cluster_map)

    # --- 4. 选出代表行并收集变体 (核心改动) ---
    final_rows = []
    for cluster_id in sorted(df_processed['cluster_id'].unique()):
        cluster_df = df_processed[df_processed['cluster_id'] == cluster_id]
        
        # 找出代表行的索引
        cluster_embeddings = np.array(cluster_df['embedding'].tolist())
        centroid = np.mean(cluster_embeddings, axis=0)
        similarities = cosine_similarity(cluster_embeddings, [centroid])
        representative_index_in_cluster = np.argmax(similarities)
        original_index = cluster_df.index[representative_index_in_cluster]
        
        # 获取代表行的完整数据（从原始df中获取，保证数据无损）
        representative_row = df.loc[original_index].to_dict()
        
        # 获取该簇中所有的原始关键词
        all_keywords_in_cluster = cluster_df[keyword_column].tolist()
        representative_keyword = representative_row[keyword_column]
        
        # 创建变体列表 (排除代表词本身)
        variants = [kw for kw in all_keywords_in_cluster if kw != representative_keyword]
        
        # 将新信息添加到行数据中
        representative_row['cluster_id'] = cluster_id
        representative_row['variants'] = variants
        
        final_rows.append(representative_row)
        
    df_final = pd.DataFrame(final_rows)
    
    return df_final

# test
if __name__ == "__main__":
    df = pd.read_csv('../data/search key/raw/search_key_combined_unclean.csv')
    # Step 1: Filter roughly medical terms (with blacklist)
    df_filtered = filter_medical_terms(df)
    print("After medical filtering:", len(df_filtered))
    # Step 2: insert back the type keywords
    df_inserted = insert_type_keyword(df_filtered)
    # Step 3: Exact Deduplication
    df_cleaned = exact_deduplication(df_inserted)
    print("After exact deduplication:", len(df_cleaned))
    # Step 4: Semantic Deduplication
    print("Loading NLP models...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    spacy_nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    print("Models loaded.")
    df_final = deduplicate_keywords(
            df=df_cleaned,
            keyword_column='search key',
            model=sbert_model,
            nlp=spacy_nlp,
            distance_threshold=0.15 
        )
    print("After deduplication:", len(df_final))
