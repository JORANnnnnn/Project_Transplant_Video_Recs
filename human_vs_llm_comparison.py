import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')

df_llm = pd.read_excel('youtube_database_export.xlsx', sheet_name='LabeledVideos')

df_g1_s1 = pd.read_excel('Videos to label.xlsx', sheet_name='1-50', skiprows=1)
df_g1_s1 = df_g1_s1.rename(columns={'Item 10': 'Item 10 the main idea is in the beginning'})
df_g1_s2 = pd.read_excel('Videos to label.xlsx', sheet_name='51-200')
df_g1 = pd.concat([df_g1_s1, df_g1_s2], ignore_index=True)
df_g1 = df_g1.drop_duplicates(subset='video_id', keep='first')

df_g2_s1 = pd.read_excel('Labeling by Leyla, Zhuoyuan.xlsx', sheet_name='1-50')
df_g2_s1 = df_g2_s1.rename(columns={'Item 10': 'Item 10 the main idea is in the beginning'})
df_g2_s2 = pd.read_excel('Labeling by Leyla, Zhuoyuan.xlsx', sheet_name='51-200')
df_g2 = pd.concat([df_g2_s1, df_g2_s2], ignore_index=True)
df_g2 = df_g2.drop_duplicates(subset='video_id', keep='first')

merged_human = df_g1.merge(df_g2, on='video_id', suffixes=('_G1', '_G2'))
merged_all = merged_human.merge(df_llm[['video_id'] + [f'PEMAT_{i}' for i in range(1,14)]], on='video_id', how='inner')

print(f'Human vs LLM Comparison')
print(f'Total videos with both human and LLM labels: {len(merged_all)}')
print()

pemat_mapping = {
    'PEMAT_1': 'PEMAT1_Purpose_Evident',
    'PEMAT_3': 'PEMAT3_Everyday_Language',
    'PEMAT_4': 'Item 4: Medical terms are used only to familiarize audience with the terms',
    'PEMAT_5': 'Item 5: The material uses the active voice (P and A/V)',
    'PEMAT_8': 'Item 8: The material breaks or "chunks" information into short sections (P and A/V)',
    'PEMAT_10': 'Item 10 the main idea is in the beginning',
    'PEMAT_11': 'Item 11: The material provides a summary (P and A/V)',
}

results = []

for llm_col, human_col in pemat_mapping.items():
    col_g1 = f'{human_col}_G1'
    col_g2 = f'{human_col}_G2'

    if col_g1 not in merged_all.columns or col_g2 not in merged_all.columns:
        continue

    human_consensus = []
    llm_labels = []

    for idx, row in merged_all.iterrows():
        g1 = row[col_g1]
        g2 = row[col_g2]
        llm = row[llm_col]

        if pd.notna(g1) and pd.notna(g2) and pd.notna(llm):
            if str(g1).strip().lower() not in ['na', 'n/a', ' na'] and str(g2).strip().lower() not in ['na', 'n/a', ' na']:
                if g1 == g2:
                    human_consensus.append(int(g1))
                    llm_labels.append(int(llm))

    if len(human_consensus) > 0:
        kappa = cohen_kappa_score(human_consensus, llm_labels)
        agreement = sum([1 for h, l in zip(human_consensus, llm_labels) if h == l]) / len(human_consensus)
        disagreements = sum([1 for h, l in zip(human_consensus, llm_labels) if h != l])

        results.append({
            'Criterion': human_col[:40],
            'N': len(human_consensus),
            'Kappa': kappa,
            'Agreement': agreement,
            'Disagreements': disagreements
        })

        print(f'{human_col[:50]}')
        print(f'  N: {len(human_consensus)}, Kappa: {kappa:.3f}, Agreement: {agreement:.1%}, Disagreements: {disagreements}')
        print()

results_df = pd.DataFrame(results)
results_df.to_csv('human_consensus_vs_llm_kappa.csv', index=False)

print('Overall Summary:')
print(f'Average Kappa: {results_df["Kappa"].mean():.3f}')
print(f'Average Agreement: {results_df["Agreement"].mean():.1%}')
print(f'Total Disagreements: {results_df["Disagreements"].sum()}')
print()
print('Saved to: human_consensus_vs_llm_kappa.csv')

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
