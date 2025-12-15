import pandas as pd

# Read the two CSVs
df1 = pd.read_csv("reddit_posts_keywords_dedup.csv")
df2 = pd.read_csv("youtube_keywords_FQA.csv")

# Combine (stack rows)
combined = pd.concat([df1, df2], ignore_index=True)

# Remove duplicates if needed
combined = combined.drop_duplicates()

# Save to a new file
combined.to_csv("keywords_reddit_FQA.csv", index=False)

print(f"Combined shape: {combined.shape}")
