#!/usr/bin/env python
# coding: utf-8

# # Google Trends Data Collection
# 
# This section downloads and consolidates **related queries** (via API) and **related topics** (from CSV files) into one dataset.
# 
# ---
# 
# ## Workflow
# 
# 1. **Initialize Pytrends**
#    - Create a `TrendReq` object with English (US), 5-year timeframe settings.
# 
# 2. **Fetch Related Queries (API)**
#    - Loop through each `base_keyword`.
#    - Call `pytrends.related_queries()`.
#    - Extract both `top` and `rising` queries.
#    - Add metadata:
#      - `base_keyword` (the seed keyword)
#      - `tag` (`top` or `rising`)
#      - `type = query`
#    - Append results to a list.
# 
# 3. **Load Related Topics (CSV)**
#    - For each `base_keyword`, read the corresponding `related_topics_{kw}.csv`.
#    - Skip the first 3 rows (metadata).
#    - Split into two parts:
#      - **TOP** topics
#      - **RISING** topics
#    - Add metadata:
#      - `base_keyword`
#      - `tag` (`top` or `rising`)
#      - `type = topic`
#    - Append results to the same list.
# 
# 4. **Combine & Save**
#    - Concatenate all query and topic dataframes.
#    - Save as `google_trends_all_uncleaned.csv`.
# 
# ---
# 
# ## Output
# - **File:** `google_trends_all_uncleaned.csv`
# - **Columns:**
#   - `search key` → query or topic text
#   - `type` → seed keyword
#   - `source` → "google trends"
# 

# In[1]:


# import necessary libraries
from pytrends.request import TrendReq
import pandas as pd
import time


# In[2]:


def get_google_trends_data(base_keywords, data_dir="../data/search key/google_trends", hl="en-US", tz=360, geo="US", timeframe="today 5-y"):
    """
    Fetch related queries (via Google Trends API) and related topics (from CSV),
    combine them into a single DataFrame.

    Parameters
    ----------
    base_keywords : list
        List of seed keywords to fetch related queries/topics for.
    data_dir : str, optional
        Directory where related_topics CSV files are stored. Default is "data".
    hl : str, optional
        Host language for Google Trends. Default "en-US".
    tz : int, optional
        Timezone offset. Default 360.
    geo : str, optional
        Geolocation parameter for Google Trends API. Default "US".
    timeframe : str, optional
        Timeframe parameter for Google Trends API. Default "today 5-y".

    Returns
    -------
    pd.DataFrame
        Combined dataframe of related queries and related topics.
    """

    pytrends = TrendReq(hl=hl, tz=tz)
    related_search_key = []

    # --- Related Queries (API) ---
    for kw in base_keywords:
        print(f"Processing (API queries): {kw}")
        try:
            pytrends.build_payload([kw], cat=0, geo=geo, timeframe=timeframe)
            time.sleep(1)  # Avoid rate limits

            related_queries = pytrends.related_queries()
            if kw in related_queries:
                if related_queries[kw]['top'] is not None:
                    df_top_q = related_queries[kw]['top'].copy()
                    df_top_q["base_keyword"] = kw
                    df_top_q['tag'] = 'top'
                    df_top_q['type_original'] = 'query'
                    df_top_q.rename(columns={'query': 'search_term'}, inplace=True)
                    related_search_key.append(df_top_q)

                if related_queries[kw]['rising'] is not None:
                    df_rising_q = related_queries[kw]['rising'].copy()
                    df_rising_q['base_keyword'] = kw
                    df_rising_q['tag'] = 'rising'
                    df_rising_q['type_original'] = 'query'
                    df_rising_q.rename(columns={'query': 'search_term'}, inplace=True)
                    related_search_key.append(df_rising_q)
        except Exception as e:
            print(f"Error processing {kw}: {e}")

    # --- Related Topics (CSV) ---
    for kw in base_keywords:
        try:
            print(f"Reading related topics CSV for: {kw}")
            N_ROWS_TO_SKIP = 3
            df = pd.read_csv(
                f"{data_dir}/related_topics_{kw}.csv",
                sep=",",
                skiprows=N_ROWS_TO_SKIP,
                header=None,
                names=["search_term", "value"],
                engine="python"
            )

            top_start = df[df["search_term"] == "TOP"].index[0] + 1
            rising_start = df[df["search_term"] == "RISING"].index[0] + 1

            # TOP
            df_top_t = df.iloc[top_start:rising_start-1].dropna()
            df_top_t["base_keyword"] = kw
            df_top_t["tag"] = "top"
            df_top_t["type_original"] = "topic"
            related_search_key.append(df_top_t)

            # RISING
            df_rising_t = df.iloc[rising_start:].dropna()
            df_rising_t["base_keyword"] = kw
            df_rising_t["tag"] = "rising"
            df_rising_t["type_original"] = "topic"
            related_search_key.append(df_rising_t)
        except FileNotFoundError:
            print(f"CSV file for {kw} not found, skipping related topics.")
        except Exception as e:
            print(f"Error processing CSV for {kw}: {e}")
        
    if not related_search_key:
        print("No data collected. Returning an empty DataFrame.")
        return pd.DataFrame(columns=['search key', 'type', 'source'])    

    # --- Combine ---
    df_all = pd.concat(related_search_key, ignore_index=True)
    df_all['source']='google trends'

    df_formatted = df_all.rename(columns={
        'search_term': 'search key',  # query or topic text
        'base_keyword': 'type'        # seed keyword
    })

    final_df = df_formatted[['search key', 'type', 'source']]
    return final_df


# In[20]:


base_keywords = ["heart transplant", "liver transplant", "kidney transplant","lung transplant", "pancreas transplant"]
df = get_google_trends_data(base_keywords, data_dir="../data/search key/google_trends")

df.to_csv("../data/search key/google_trends/google_trends_all_uncleaned.csv", index=False)
print("✅ Saved data to google_trends_all_uncleaned.csv")


# # Data Cleaning & Deduplication
# 
# This section processes the raw Google Trends dataset into a **clean, deduplicated, and medical-relevant dataset**.
# 
# ---
# 
# ## Workflow Overview
# 
# The cleaning process includes **four steps**:
# 
# 1. **Zero-Shot Classification Filtering**  
#    - Use a pre-trained transformer (`facebook/bart-large-mnli`) for zero-shot classification.  
#    - Keep only terms whose top predicted label is in:  
#      - `["medical", "health", "disease"]`
# 
# 2. **Combine 'type' keywords back into the search keys**
#    - for example, for search key `"surgery"`, combine it with the type `"heart transplant"` into `"heart transplant surgery"` to be more specific on the search terms
#    - for those terms already have specific type in them, just skip them.
# 
# 3. **Exact Deduplication**  
#    - Merge duplicate `search_term` entries.  
#    - Combine associated `base_keywords` into a list.   
# 
# 4. **Fuzzy Deduplication**  
#    - Apply fuzzy string matching (`fuzz.ratio`) to detect and merge semantically similar terms.  
#    - Ensures the final dataset is **unique, medical-relevant, and cleaned**.
#    - Final dataset contains:  
#    - `search_term` → main keyword kept  
#    - `type` → list of associated base keywords  
#    - `source` → `"google trends"`  
#    - `variants` → list of merged similar terms 

# In[24]:


import pandas as pd
from transformers import pipeline
from fuzzywuzzy import fuzz
import re


# In[25]:


# combine type keywords to the search term


# In[66]:


# 1. Zero-shot classification filtering out irrelavate terms

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


# 4. Fuzzy Deduplication
def fuzzy_deduplication(df, threshold=90):
    """
    Perform fuzzy deduplication based on string similarity.
    """
    groups = []  # 每个 group: {"search_term": canonical, "variants": [...], "type": [...], "source": ...}

    for _, row in df.iterrows():
        term = str(row['search key'])
        term = term.strip()
        types = row['type'] if isinstance(row['type'], list) else [row['type']]
        source = row.get('source')

        assigned = False
        for g in groups:
            # 使用 partial_ratio 进行相似度判断
            if fuzz.partial_ratio(term, g['search key']) >= threshold:
                g['variants'].append(term)
                g['type'].extend(types)
                assigned = True
                break

        if not assigned:
            groups.append({
                "search key": term,
                "variants": [term],
                "type": types.copy(),
                "source": source
            })

    # 构建结果 DataFrame，去重 types 与 variants 并保留顺序
    rows = []
    for g in groups:
        # 去重并保留插入顺序
        unique_types = list(dict.fromkeys(g['type']))
        unique_variants = list(dict.fromkeys(g['variants']))
        rows.append({
            "search key": g['search key'],
            "type": unique_types,
            "source": g['source'],
            "variants": unique_variants
        })

    return pd.DataFrame(rows).reset_index(drop=True)


# In[57]:


from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import AgglomerativeClustering
def cluster_by_semantic_similarity(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['search_term'].tolist())

    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0)
    labels = clustering.fit_predict(embeddings)

    df['cluster'] = labels

    def split_cluster_terms(terms):
        return terms[0], terms[1:]

    df_clustered = df.groupby('cluster').agg({
        'search_term': list,
        'base_keyword': lambda x: list(set(sum([k if isinstance(k, list) else [k] for k in x], [])))
    }).reset_index()

    df_clustered[['main_term', 'other_terms']] = df_clustered['search_term'].apply(
        lambda x: pd.Series(split_cluster_terms(x))
    )

    df_clustered = df_clustered.drop(columns=['search_term'])
    return df_clustered.reset_index(drop=True)


# In[47]:


# Step 1: Filter roughly medical terms (with blacklist)
df_filtered = filter_medical_terms(df)
print("After medical filtering:", len(df_filtered))



# In[67]:


# Step 2: insert back the type keywords
df_inserted = insert_type_keyword(df_filtered)

# Step 3: Exact Deduplication
df_cleaned = exact_deduplication(df_inserted)
print("After exact deduplication:", len(df_cleaned))


# In[69]:


# Step 4: Fuzzy Deduplication
df_final = fuzzy_deduplication(df_cleaned, threshold=90)
print("After fuzzy deduplication:", len(df_final))

# Save final dataset
df_final.to_csv("../data/search key/google_trends/google_trends_cleaned.csv", index=False)
print("✅ Saved to google_trends_cleaned.csv")


# In[ ]:




