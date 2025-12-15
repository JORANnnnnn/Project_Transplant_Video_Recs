import pandas as pd

# Load your CSV
df = pd.read_csv("reddit_posts_keywords.csv")

# Drop duplicate rows (based on both columns)
df_unique = df.drop_duplicates()

# Or, if you want to remove duplicates only by keyword, keep first occurrence:
# df_unique = df.drop_duplicates(subset=["keywords"])

# Save cleaned file
df_unique.to_csv("reddit_posts_keywords_dedup.csv", index=False)

print(f"Original rows: {len(df)}")
print(f"Unique rows: {len(df_unique)}")
print("Saved to reddit_posts_keywords_dedup.csv")
