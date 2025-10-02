import pandas as pd

# Define the 25 YouTube-ready keywords
keywords = [
    "transplant diet restrictions",
    "foods to avoid after transplant",
    "immunosuppressant side effects",
    "nausea pain tremor management",
    "daily precautions immunocompromised",
    "transplant waitlist process explained",
    "transplant eligibility age conditions",
    "common transplant complications",
    "signs of transplant rejection",
    "transplant follow up tests",
    "over the counter meds after transplant",
    "transplant recovery timeline",
    "transplant insurance costs",
    "psychological effects after transplant",
    "transplant patient depression anxiety",
    "travel tips after transplant",
    "infection prevention transplant patients",
    "caregiver tips transplant recovery",
    "switching transplant hospital",
    "organ compatibility and matching",
    "exercise after transplant",
    "long term anti rejection side effects",
    "symptoms of organ rejection",
    "safe eating after transplant",
    "sun exposure skin protection transplant"
]

# Create DataFrame
df = pd.DataFrame({
    "keywords": keywords,
    "transplant_type": ["general"] * len(keywords)
})

# Save to CSV
output_file = "youtube_keywords_FQA.csv"
df.to_csv(output_file, index=False)

output_file
