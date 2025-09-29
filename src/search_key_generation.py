import google_trends_search_key_generation
import google_autocom_search_key
import search_key_clean_deduplicate
import sentence_transformers
import spacy
import pandas as pd
import os
from datetime import datetime

def merge_all_csv_files():
    """
    合并所有生成的CSV文件到一个文件中，并删除原始文件
    """
    raw_dir = "../data/search key/raw"
    all_dataframes = []
    
    # 获取所有需要合并的CSV文件
    csv_files = []
    
    # 添加google_complete_*.csv文件
    google_complete_files = [f for f in os.listdir(raw_dir) if f.startswith('google_complete_') and f.endswith('.csv')]
    csv_files.extend(google_complete_files)
    
    # 添加google_trends_all_uncleaned.csv文件
    trends_file = "google_trends_all_uncleaned.csv"
    if os.path.exists(os.path.join(raw_dir, trends_file)):
        csv_files.append(trends_file)
    
    print(f"找到 {len(csv_files)} 个CSV文件需要合并:")
    for file in csv_files:
        print(f"  - {file}")
    
    # 读取并合并所有CSV文件
    for file in csv_files:
        filepath = os.path.join(raw_dir, file)
        try:
            df = pd.read_csv(filepath)
            all_dataframes.append(df)
            print(f"✓ 已读取: {file} ({len(df)} 行)")
        except Exception as e:
            print(f"✗ 读取失败: {file} - {e}")
    
    if all_dataframes:
        # 合并所有数据框
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        
        # 去重
        merged_df = merged_df.drop_duplicates(subset=['search key'], keep='first')
        
        # 保存合并后的文件
        output_file = os.path.join(raw_dir, "search_key_combined_unclean.csv")
        merged_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\n合并完成!")
        print(f"总记录数: {len(merged_df)}")
        print(f"保存到: {output_file}")
        
        # 删除原始文件
        print(f"\n删除原始文件:")
        for file in csv_files:
            filepath = os.path.join(raw_dir, file)
            try:
                os.remove(filepath)
                print(f"✓ 已删除: {file}")
            except Exception as e:
                print(f"✗ 删除失败: {file} - {e}")
        
        return output_file
    else:
        print("没有找到可合并的CSV文件")
        return None

if __name__ == "__main__":
    # #1: get search keys from google autocom and google trends; merge them into one csv file
    # for type in ("heart transplant", "kidney transplant", "liver transplant", "pancreas transplant", "heart-lung transplant","lung transplant"):
    #     results = google_autocom_search_key.get_suggestions_combined(type, max_workers=8, top_n=15)
    #     for i, suggestion in enumerate(results[:20], 1):
    #         print(f"{i:2d}. {suggestion}")
    #     if len(results) > 20:
    #         print(f"... and {len(results) - 20} more results")
    #     google_autocom_search_key.save_to_csv(results, type=type)
    
    # base_keywords = ["heart transplant", "liver transplant", "kidney transplant","lung transplant", "pancreas transplant"]
    # df = google_trends_search_key_generation.get_google_trends_data(base_keywords, data_dir="../data/search key/google_trends")

    # df.to_csv("../data/search key/raw/google_trends_all_uncleaned.csv", index=False)
    
    # # 合并所有生成的CSV文件
    # print("\n" + "="*50)
    # print("开始合并所有CSV文件...")
    # merge_all_csv_files()
    
    
    
    # 2: clean and deduplicate the search keys
    df = pd.read_csv('../data/search key/raw/search_key_combined_unclean.csv')
    
    # Step 1: Filter roughly medical terms (with blacklist)
    df_filtered = search_key_clean_deduplicate.filter_medical_terms(df)
    print("After medical filtering:", len(df_filtered))
    
    # Step 2: insert back the type keywords
    df_inserted = search_key_clean_deduplicate.insert_type_keyword(df_filtered)
    
    # Step 3: Exact Deduplication
    df_cleaned = search_key_clean_deduplicate.exact_deduplication(df_inserted)
    print("After exact deduplication:", len(df_cleaned))
    
    # Step 4: Semantic Deduplication
    print("Loading NLP models...")
    sbert_model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
    spacy_nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    print("Models loaded.")
    df_final = search_key_clean_deduplicate.deduplicate_keywords(
            df=df_cleaned,
            keyword_column='search key',
            model=sbert_model,
            nlp=spacy_nlp,
            distance_threshold=0.15 
        )
    print("After deduplication:", len(df_final))
    df_final.to_csv('../data/search key/raw/search_key_combined_cleaned.csv', index=False)
    print("✅ Saved to search_key_combined_cleaned.csv")