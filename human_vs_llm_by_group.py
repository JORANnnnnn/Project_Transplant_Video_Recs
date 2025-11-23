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

print('=' * 80)
print('LLM vs HUMAN COMPARISON: THREE APPROACHES')
print('=' * 80)
print()

pemat_mapping = {
    'PEMAT_1': 'PEMAT1_Purpose_Evident',
    'PEMAT_2': 'PEMAT3_Everyday_Language',
    'PEMAT_3': 'Item 4: Medical terms are used only to familiarize audience with the terms',
    'PEMAT_4': 'Item 5: The material uses the active voice (P and A/V)',
    'PEMAT_5': 'Item 8: The material breaks or "chunks" information into short sections (P and A/V)',
    'PEMAT_8': 'Item 11: The material provides a summary (P and A/V)',
    'PEMAT_7': 'Item 10 the main idea is in the beginning',
}

results_consensus = []
results_g1 = []
results_g2 = []

for llm_col, human_col in pemat_mapping.items():
    col_g1 = f'{human_col}_G1'
    col_g2 = f'{human_col}_G2'

    if col_g1 not in merged_all.columns or col_g2 not in merged_all.columns:
        continue

    # Consensus approach
    human_consensus = []
    llm_consensus = []

    # Group 1 only
    human_g1_only = []
    llm_g1_only = []

    # Group 2 only
    human_g2_only = []
    llm_g2_only = []

    for idx, row in merged_all.iterrows():
        g1 = row[col_g1]
        g2 = row[col_g2]
        llm = row[llm_col]

        # Consensus
        if pd.notna(g1) and pd.notna(g2) and pd.notna(llm):
            if str(g1).strip().lower() not in ['na', 'n/a', ' na'] and str(g2).strip().lower() not in ['na', 'n/a', ' na']:
                if g1 == g2:
                    human_consensus.append(int(g1))
                    llm_consensus.append(int(llm))

        # Group 1
        if pd.notna(g1) and pd.notna(llm):
            if str(g1).strip().lower() not in ['na', 'n/a', ' na']:
                human_g1_only.append(int(g1))
                llm_g1_only.append(int(llm))

        # Group 2
        if pd.notna(g2) and pd.notna(llm):
            if str(g2).strip().lower() not in ['na', 'n/a', ' na']:
                human_g2_only.append(int(g2))
                llm_g2_only.append(int(llm))

    # Calculate metrics
    if len(human_consensus) > 0:
        results_consensus.append({
            'Criterion': human_col[:35],
            'N': len(human_consensus),
            'Kappa': cohen_kappa_score(human_consensus, llm_consensus),
            'Agreement': sum([1 for h, l in zip(human_consensus, llm_consensus) if h == l]) / len(human_consensus),
        })

    if len(human_g1_only) > 0:
        results_g1.append({
            'Criterion': human_col[:35],
            'N': len(human_g1_only),
            'Kappa': cohen_kappa_score(human_g1_only, llm_g1_only),
            'Agreement': sum([1 for h, l in zip(human_g1_only, llm_g1_only) if h == l]) / len(human_g1_only),
        })

    if len(human_g2_only) > 0:
        results_g2.append({
            'Criterion': human_col[:35],
            'N': len(human_g2_only),
            'Kappa': cohen_kappa_score(human_g2_only, llm_g2_only),
            'Agreement': sum([1 for h, l in zip(human_g2_only, llm_g2_only) if h == l]) / len(human_g2_only),
        })

# Convert to DataFrames
df_consensus = pd.DataFrame(results_consensus)
df_g1 = pd.DataFrame(results_g1)
df_g2 = pd.DataFrame(results_g2)

# Display comparison table
print('Criterion                           | Consensus |  Group 1  |  Group 2  |')
print('                                    |  N  Kappa |  N  Kappa |  N  Kappa |')
print('-' * 80)

for i in range(len(df_consensus)):
    crit = df_consensus.iloc[i]['Criterion']

    n_c = df_consensus.iloc[i]['N']
    k_c = df_consensus.iloc[i]['Kappa']

    n_1 = df_g1.iloc[i]['N']
    k_1 = df_g1.iloc[i]['Kappa']

    n_2 = df_g2.iloc[i]['N']
    k_2 = df_g2.iloc[i]['Kappa']

    print(f'{crit:35} | {n_c:3d} {k_c:5.3f} | {n_1:3d} {k_1:5.3f} | {n_2:3d} {k_2:5.3f} |')

print('-' * 80)
print(f'{"AVERAGE":35} | {df_consensus["N"].mean():3.0f} {df_consensus["Kappa"].mean():5.3f} | {df_g1["N"].mean():3.0f} {df_g1["Kappa"].mean():5.3f} | {df_g2["N"].mean():3.0f} {df_g2["Kappa"].mean():5.3f} |')
print()

print('=' * 80)
print('AGREEMENT COUNTS (Agree/Total per criterion)')
print('=' * 80)
print()
print('Criterion                           |  Consensus  |   Group 1   |   Group 2   |')
print('-' * 80)

for i in range(len(df_consensus)):
    crit = df_consensus.iloc[i]['Criterion']

    n_c = df_consensus.iloc[i]['N']
    a_c = df_consensus.iloc[i]['Agreement']
    agree_c = int(a_c * n_c)

    n_1 = df_g1.iloc[i]['N']
    a_1 = df_g1.iloc[i]['Agreement']
    agree_1 = int(a_1 * n_1)

    n_2 = df_g2.iloc[i]['N']
    a_2 = df_g2.iloc[i]['Agreement']
    agree_2 = int(a_2 * n_2)

    print(f'{crit:35} | {agree_c:3d}/{n_c:3d} ({a_c:.1%}) | {agree_1:3d}/{n_1:3d} ({a_1:.1%}) | {agree_2:3d}/{n_2:3d} ({a_2:.1%}) |')

print('-' * 80)

total_agree_c = sum([int(df_consensus.iloc[i]['Agreement'] * df_consensus.iloc[i]['N']) for i in range(len(df_consensus))])
total_n_c = df_consensus['N'].sum()

total_agree_1 = sum([int(df_g1.iloc[i]['Agreement'] * df_g1.iloc[i]['N']) for i in range(len(df_g1))])
total_n_1 = df_g1['N'].sum()

total_agree_2 = sum([int(df_g2.iloc[i]['Agreement'] * df_g2.iloc[i]['N']) for i in range(len(df_g2))])
total_n_2 = df_g2['N'].sum()

print(f'{"TOTAL":35} | {total_agree_c:3d}/{total_n_c:3d} ({total_agree_c/total_n_c:.1%}) | {total_agree_1:3d}/{total_n_1:3d} ({total_agree_1/total_n_1:.1%}) | {total_agree_2:3d}/{total_n_2:3d} ({total_agree_2/total_n_2:.1%}) |')
print()

# Add agreement columns (Agree/Total format)
df_consensus['Agreement_Count'] = df_consensus.apply(lambda row: f"{int(row['Agreement'] * row['N'])}/{row['N']}", axis=1)
df_g1['Agreement_Count'] = df_g1.apply(lambda row: f"{int(row['Agreement'] * row['N'])}/{row['N']}", axis=1)
df_g2['Agreement_Count'] = df_g2.apply(lambda row: f"{int(row['Agreement'] * row['N'])}/{row['N']}", axis=1)

# Reorder columns to put Agreement_Count after Agreement
df_consensus = df_consensus[['Criterion', 'N', 'Kappa', 'Agreement', 'Agreement_Count']]
df_g1 = df_g1[['Criterion', 'N', 'Kappa', 'Agreement', 'Agreement_Count']]
df_g2 = df_g2[['Criterion', 'N', 'Kappa', 'Agreement', 'Agreement_Count']]

# Save results to one Excel file with multiple sheets
with pd.ExcelWriter('llm_vs_human_comparison.xlsx', engine='openpyxl') as writer:
    df_consensus.to_excel(writer, sheet_name='Consensus', index=False)
    df_g1.to_excel(writer, sheet_name='Group1', index=False)
    df_g2.to_excel(writer, sheet_name='Group2', index=False)

print('Saved: llm_vs_human_comparison.xlsx')
print('  - Sheet: Consensus (LLM vs human consensus)')
print('  - Sheet: Group1 (LLM vs Group 1 - Zihan + Vinayak)')
print('  - Sheet: Group2 (LLM vs Group 2 - Leyla + Zhuoyuan)')

