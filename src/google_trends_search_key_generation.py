from pytrends.request import TrendReq
import pandas as pd
import time


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

if __name__ == "__main__":
    base_keywords = ["heart transplant", "liver transplant", "kidney transplant","lung transplant", "pancreas transplant"]
    df = get_google_trends_data(base_keywords, data_dir="../data/search key/google_trends")

    df.to_csv("../data/search key/raw/google_trends_all_uncleaned.csv", index=False)
    print("✅ Saved data to google_trends_all_uncleaned.csv")