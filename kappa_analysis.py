import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


df_group1_s1 = pd.read_excel('Videos to label.xlsx', sheet_name='1-50', skiprows=1)
df_group1_s1 = df_group1_s1.rename(columns={'Item 10': 'Item 10 the main idea is in the beginning'})
df_group1_s2 = pd.read_excel('Videos to label.xlsx', sheet_name='51-200')
df_group1 = pd.concat([df_group1_s1, df_group1_s2], ignore_index=True)
df_group1 = df_group1.drop_duplicates(subset='video_id', keep='first')

df_group2_s1 = pd.read_excel('Labeling by Leyla, Zhuoyuan.xlsx', sheet_name='1-50')
df_group2_s1 = df_group2_s1.rename(columns={'Item 10': 'Item 10 the main idea is in the beginning'})
df_group2_s2 = pd.read_excel('Labeling by Leyla, Zhuoyuan.xlsx', sheet_name='51-200')
df_group2 = pd.concat([df_group2_s1, df_group2_s2], ignore_index=True)
df_group2 = df_group2.drop_duplicates(subset='video_id', keep='first')

merged = df_group1.merge(df_group2, on='video_id', suffixes=('_G1', '_G2'))

# PEMAT
pemat_cols = [
    'PEMAT1_Purpose_Evident',
    'PEMAT3_Everyday_Language',
    'Item 4: Medical terms are used only to familiarize audience with the terms',
    'Item 5: The material uses the active voice (P and A/V)',
    'Item 8: The material breaks or "chunks" information into short sections (P and A/V)',
    "Item 9: The material's sections have informative headers (P and A/V)",
    'Item 10 the main idea is in the beginning',
    'Item 11: The material provides a summary (P and A/V)',
    'Item 13: Text on the screen is easy to read (A/V)',
    'Item 14: The material allows the user to hear the words clearly (e.g., not too fast, not garbled) (A/V)',
    'Item 18: The material uses illustrations and photographs that are clear and uncluttered (P and A/V)'
]

print('Kappa Analysis: Group 1 (Zihan+Vinayak) vs Group 2 (Leyla+Zhuoyuan)')
print(f'Total videos compared: {len(merged)}')
print()

results = []

for col in pemat_cols:
    col_g1 = f'{col}_G1'
    col_g2 = f'{col}_G2'

    if col_g1 not in merged.columns or col_g2 not in merged.columns:
        continue

    # Filter out rows where both have valid labels
    valid = merged[[col_g1, col_g2]].dropna()

    if len(valid) == 0:
        continue

    g1_labels = valid[col_g1].replace({'NA': np.nan, 'na': np.nan, ' na': np.nan, ' NA': np.nan, 'N/A': np.nan})
    g2_labels = valid[col_g2].replace({'NA': np.nan, 'na': np.nan, ' na': np.nan, ' NA': np.nan, 'N/A': np.nan})
    g1_labels = g1_labels.dropna()
    g2_labels = g2_labels.dropna()

    # Get common indices
    common_idx = g1_labels.index.intersection(g2_labels.index)
    g1_labels = g1_labels.loc[common_idx].astype(int).values
    g2_labels = g2_labels.loc[common_idx].astype(int).values

    if len(g1_labels) == 0:
        continue

    # kappa
    kappa = cohen_kappa_score(g1_labels, g2_labels)

  
    agreement = (g1_labels == g2_labels).sum() / len(g1_labels)

  
    disagreements = (g1_labels != g2_labels).sum()

    results.append({
        'Criterion': col[:50],
        'N': len(valid),
        'Kappa': kappa,
        'Agreement': agreement,
        'Disagreements': disagreements
    })

    print(f'{col[:50]}')
    print(f'  N: {len(valid)}, Kappa: {kappa:.3f}, Agreement: {agreement:.1%}, Disagreements: {disagreements}')
    print()

# Summary
results_df = pd.DataFrame(results)
results_df.to_csv('kappa_results.csv', index=False)

print('Overall Summary:')
print(f'Average Kappa: {results_df["Kappa"].mean():.3f}')
print(f'Average Agreement: {results_df["Agreement"].mean():.1%}')
print(f'Total Disagreements: {results_df["Disagreements"].sum()}')
print()


avg_kappa = results_df['Kappa'].mean()
if avg_kappa > 0.80:
    interpretation = 'Excellent agreement'
elif avg_kappa > 0.60:
    interpretation = 'Substantial agreement'
elif avg_kappa > 0.40:
    interpretation = 'Moderate agreement'
elif avg_kappa > 0.20:
    interpretation = 'Fair agreement'
else:
    interpretation = 'Poor agreement'

print(f'Interpretation: {interpretation}')
