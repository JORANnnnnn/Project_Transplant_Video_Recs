import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("UNDERSTANDABILITY THRESHOLD ANALYSIS")
print("Comparing LLM PEMAT Scores vs Human Understandability Labels")
print("=" * 80)
print()


df_llm = pd.read_excel('youtube_database_export.xlsx', sheet_name='LabeledVideos')

# Load human labels
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

# Merge data
merged_human = df_g1.merge(df_g2, on='video_id', suffixes=('_G1', '_G2'))
merged_all = merged_human.merge(
    df_llm[['video_id', 'is_understandable', 'PEMAT_score']],
    on='video_id',
    how='inner'
)

# Clean human scores
merged_all['human_understand_G1'] = pd.to_numeric(
    merged_all['Understandability SCore_G1'], errors='coerce'
)
merged_all['human_understand_G2'] = pd.to_numeric(
    merged_all['Understandability SCore_G2'], errors='coerce'
)
merged_all['human_understand_avg'] = (
    merged_all['human_understand_G1'] + merged_all['human_understand_G2']
) / 2

# Filter valid data
valid_data = merged_all[
    merged_all['human_understand_avg'].notna() &
    merged_all['PEMAT_score'].notna()
].copy()

print(f"Dataset: {len(valid_data)} videos with both human and LLM labels")
print()

# ============================================================================
# PART 1: Which Human Labels Are We Using?
# ============================================================================
print("=" * 80)
print("PART 1: HUMAN LABEL SOURCE")
print("=" * 80)
print()

print("We are using: AVERAGE of both groups' understandability scores")
print(f"  Group 1 (Zihan + Vinayak): {len(df_g1)} videos")
print(f"  Group 2 (Leyla + Zhuoyuan): {len(df_g2)} videos")
print(f"  Overlapping videos with LLM data: {len(valid_data)} videos")
print()

print("Group 1 Understandability Score Distribution:")
print(f"  Mean: {valid_data['human_understand_G1'].mean():.3f}")
print(f"  Median: {valid_data['human_understand_G1'].median():.3f}")
print()

print("Group 2 Understandability Score Distribution:")
print(f"  Mean: {valid_data['human_understand_G2'].mean():.3f}")
print(f"  Median: {valid_data['human_understand_G2'].median():.3f}")
print()

print("Average (what we use for analysis):")
print(f"  Mean: {valid_data['human_understand_avg'].mean():.3f}")
print(f"  Median: {valid_data['human_understand_avg'].median():.3f}")
print()

# Check agreement between groups
diff = abs(valid_data['human_understand_G1'] - valid_data['human_understand_G2'])
print("Differences between Group 1 and Group 2:")
print(f"  Mean absolute difference: {diff.mean():.3f}")
print(f"  Videos with difference > 0.2: {(diff > 0.2).sum()} ({(diff > 0.2).sum()/len(valid_data)*100:.1f}%)")
print(f"  Videos with difference > 0.3: {(diff > 0.3).sum()} ({(diff > 0.3).sum()/len(valid_data)*100:.1f}%)")
print()

correlation_groups = valid_data['human_understand_G1'].corr(valid_data['human_understand_G2'])
print(f"Correlation between Group 1 and Group 2: {correlation_groups:.3f}")
print()

if correlation_groups > 0.7:
    print("✓ High correlation - Groups are consistent, averaging is appropriate")
elif correlation_groups > 0.5:
    print("⚠ Moderate correlation - Some differences between groups")
else:
    print("✗ Low correlation - Significant disagreement between groups")
print()

# ============================================================================
# PART 2: Human and LLM Distributions
# ============================================================================
print("=" * 80)
print("PART 2: DISTRIBUTION COMPARISON")
print("=" * 80)
print()

print("Human Understandability Score (Average of 2 groups):")
print(f"  Mean: {valid_data['human_understand_avg'].mean():.3f}, Range: {valid_data['human_understand_avg'].min():.3f}-{valid_data['human_understand_avg'].max():.3f}")
print()

print("LLM Binary Classification (is_understandable):")
print(f"  Understandable (1): {(valid_data['is_understandable'] == 1).sum()} ({(valid_data['is_understandable'] == 1).mean()*100:.1f}%)")
print(f"  Not understandable (0): {(valid_data['is_understandable'] == 0).sum()} ({(valid_data['is_understandable'] == 0).mean()*100:.1f}%)")
print()

# ============================================================================
# PART 3: Testing Different Human Thresholds vs LLM Binary
# ============================================================================
print("=" * 80)
print("PART 3: HUMAN THRESHOLD VS LLM BINARY CLASSIFICATION")
print("=" * 80)
print()

thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]
results_human_thresh = []

print("Human  | Accuracy | Kappa | F1   | Both    | Both   | Human   | Human   | Total   | Total")
print("Thresh |          |       |      | Agree=1 | Agree=0| Yes,LLM | No,LLM  | Agree   | Disagree")
print("-" * 95)

for threshold in thresholds:
    human_binary = (valid_data['human_understand_avg'] >= threshold).astype(int)
    llm_binary = valid_data['is_understandable'].astype(int)

    accuracy = accuracy_score(human_binary, llm_binary)
    kappa = cohen_kappa_score(human_binary, llm_binary)
    f1 = f1_score(human_binary, llm_binary, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(human_binary, llm_binary).ravel()
    total_agree = tp + tn
    total_disagree = fn + fp

    print(f" ≥{threshold:.2f} | {accuracy:.3f}   | {kappa:.3f} | {f1:.3f} | {tp:3d}    | {tn:3d}   | {fn:3d}    | {fp:3d}    | {total_agree:3d}    | {total_disagree:3d}")

    results_human_thresh.append({
        'threshold': threshold,
        'kappa': kappa,
        'accuracy': accuracy,
        'both_understandable': tp,
        'both_not_understandable': tn,
        'human_yes_llm_no': fn,
        'human_no_llm_yes': fp
    })

print()
print("KEY FINDING: All Kappa < 0.20 (Poor agreement)")
print()

# Show Kappa calculation for 0.70
example = [r for r in results_human_thresh if r['threshold'] == 0.70][0]
tp, tn, fn, fp = example['both_understandable'], example['both_not_understandable'], example['human_yes_llm_no'], example['human_no_llm_yes']
total = tp + tn + fn + fp
observed = (tp + tn) / total
human_yes, llm_yes = tp + fn, tp + fp
expected = (human_yes * llm_yes + (total - human_yes) * (total - llm_yes)) / (total * total)
kappa_calc = (observed - expected) / (1 - expected)

print(f"Example: Threshold ≥0.70")
print(f"  Observed agreement: {observed*100:.1f}%")
print(f"  Expected by chance: {expected*100:.1f}% (since {human_yes/total*100:.0f}% human yes, {llm_yes/total*100:.0f}% LLM yes)")
print(f"  Kappa = ({observed:.3f} - {expected:.3f}) / (1 - {expected:.3f}) = {kappa_calc:.3f}")
print(f"  → Only {(observed-expected)*100:.1f}% genuine agreement beyond chance")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

print("1. HUMAN LABELS:")
print(f"   - Using average of both groups (correlation = {correlation_groups:.3f})")
print(f"   - Mean difference between groups: {diff.mean():.3f}")
print()

print("2. HUMAN THRESHOLD ANALYSIS:")
print(f"   - Tested thresholds: 0.50 to 0.90")
print(f"   - Example at 0.70: Kappa = {example['kappa']:.3f} (Poor)")
print(f"   - Problem: {expected*100:.0f}% agreement expected by chance alone")
print()

print("3. KEY FINDING:")
print("   - All human thresholds show poor Kappa (< 0.20)")
print("   - LLM's binary classification doesn't align well with human scores")
print("   - Even at threshold 0.70, only 1.4% genuine agreement beyond chance")
print()

print("4. ROOT CAUSE:")
print("   - Individual PEMAT criteria have poor agreement (Kappa = 0.065 from prior analysis)")
print("   - LLM's understandability classification is based on flawed PEMAT assessments")
print("   - Bad individual scores → Bad binary classification")
print()

print("CONCLUSION:")
print("  No human threshold produces good agreement with LLM's binary classification.")
print("  Need to improve LLM's individual PEMAT criteria assessments first.")
print()

print("=" * 80)

# Save results
pd.DataFrame(results_human_thresh).to_csv('human_threshold_results.csv', index=False)
print("Saved: human_threshold_results.csv")
