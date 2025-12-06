# ============================================================================
# 3. SCALE ANALYSIS - Feature Ranges and Scales (IMPROVED VERSION)
# ============================================================================
print("=" * 80)
print("3. SCALE ANALYSIS - Feature Ranges and Scales")
print("=" * 80)

scale_info = []

# Ensure numeric_cols is defined
if 'numeric_cols' not in locals():
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"📊 Found {len(numeric_cols)} numeric columns")

# Process each numeric column with error handling
for col in numeric_cols:
    try:
        # Check if column exists in dataframe
        if col not in df.columns:
            continue
            
        col_data = df[col].dropna()
        
        if len(col_data) > 0:
            min_val = col_data.min()
            max_val = col_data.max()
            mean_val = col_data.mean()
            std_val = col_data.std()
            median_val = col_data.median()
            q25 = col_data.quantile(0.25)
            q75 = col_data.quantile(0.75)
            iqr = q75 - q25
            
            # Improved scale classification
            max_abs = max(abs(max_val), abs(min_val))
            if max_abs > 1000:
                scale_type = 'Large'
            elif max_abs > 10:
                scale_type = 'Medium'
            else:
                scale_type = 'Small'
            
            # Calculate coefficient of variation (CV) for scale consistency
            cv = (std_val / abs(mean_val)) if mean_val != 0 else 999
            
            scale_info.append({
                'Feature': col,
                'Min': min_val,
                'Max': max_val,
                'Range': max_val - min_val,
                'Mean': mean_val,
                'Median': median_val,
                'Std': std_val,
                'IQR': iqr,
                'CV': cv if cv != 999 else 999,
                'Scale_Type': scale_type
            })
    except Exception as e:
        print(f"⚠️  Error processing column {col}: {e}")
        continue

if len(scale_info) == 0:
    print("⚠️  No numeric features found to analyze!")
else:
    scale_df = pd.DataFrame(scale_info).sort_values('Range', ascending=False)
    
    print(f"\n📊 Feature Scale Summary (Top 20 by range):")
    display_cols = ['Feature', 'Min', 'Max', 'Range', 'Mean', 'Std', 'Scale_Type']
    print(scale_df[display_cols].head(20).to_string(index=False))
    
    # Enhanced visualizations
    fig = plt.figure(figsize=(20, 14))
    
    # Plot 1: Range of features (with value labels)
    ax1 = plt.subplot(2, 3, 1)
    top_20 = scale_df.head(20)
    bars1 = ax1.barh(range(len(top_20)), top_20['Range'], color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_yticks(range(len(top_20)))
    ax1.set_yticklabels([f[:25] for f in top_20['Feature']], fontsize=8)
    ax1.set_xlabel('Range (Max - Min)', fontsize=11)
    ax1.set_title('Top 20 Features by Range', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Standard deviation
    ax2 = plt.subplot(2, 3, 2)
    bars2 = ax2.barh(range(len(top_20)), top_20['Std'], color='coral', alpha=0.7, edgecolor='black')
    ax2.set_yticks(range(len(top_20)))
    ax2.set_yticklabels([f[:25] for f in top_20['Feature']], fontsize=8)
    ax2.set_xlabel('Standard Deviation', fontsize=11)
    ax2.set_title('Top 20 Features by Std Dev', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Plot 3: Scale type distribution (enhanced)
    ax3 = plt.subplot(2, 3, 3)
    scale_type_counts = scale_df['Scale_Type'].value_counts()
    colors_map = {'Large': 'lightcoral', 'Medium': 'lightblue', 'Small': 'lightgreen'}
    colors_list = [colors_map.get(st, 'gray') for st in scale_type_counts.index]
    wedges, texts, autotexts = ax3.pie(scale_type_counts.values, labels=scale_type_counts.index, 
                                       autopct='%1.1f%%', colors=colors_list, startangle=90)
    ax3.set_title('Feature Scale Type Distribution', fontsize=12, fontweight='bold')
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    # Plot 4: Range vs Std scatter (with scale type colors)
    ax4 = plt.subplot(2, 3, 4)
    for scale_type in scale_df['Scale_Type'].unique():
        subset = scale_df[scale_df['Scale_Type'] == scale_type]
        ax4.scatter(subset['Range'], subset['Std'], alpha=0.6, s=60, 
                   c=colors_map.get(scale_type, 'gray'), label=scale_type, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Range', fontsize=11)
    ax4.set_ylabel('Standard Deviation', fontsize=11)
    ax4.set_title('Range vs Standard Deviation (by Scale Type)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    
    # Plot 5: Log scale range
    ax5 = plt.subplot(2, 3, 5)
    top_30 = scale_df.head(30)
    bars5 = ax5.barh(range(len(top_30)), np.log1p(top_30['Range']), color='teal', alpha=0.7, edgecolor='black')
    ax5.set_yticks(range(len(top_30)))
    ax5.set_yticklabels([f[:20] for f in top_30['Feature']], fontsize=7)
    ax5.set_xlabel('Log(Range + 1)', fontsize=11)
    ax5.set_title('Top 30 Features (Log Scale)', fontsize=12, fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)
    
    # Plot 6: Coefficient of Variation (CV)
    ax6 = plt.subplot(2, 3, 6)
    cv_data = scale_df[scale_df['CV'] < 100].head(20)
    if len(cv_data) > 0:
        bars6 = ax6.barh(range(len(cv_data)), cv_data['CV'], color='purple', alpha=0.7, edgecolor='black')
        ax6.set_yticks(range(len(cv_data)))
        ax6.set_yticklabels([f[:25] for f in cv_data['Feature']], fontsize=8)
        ax6.set_xlabel('Coefficient of Variation (CV)', fontsize=11)
        ax6.set_title('Top 20 Features by CV (Relative Variability)', fontsize=12, fontweight='bold')
        ax6.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Identify features with very different scales
    large_scale = scale_df[scale_df['Scale_Type'] == 'Large']
    small_scale = scale_df[scale_df['Scale_Type'] == 'Small']
    medium_scale = scale_df[scale_df['Scale_Type'] == 'Medium']
    
    print(f"\n📊 Scale Type Summary:")
    print(f"   Large scale (|value| > 1000): {len(large_scale)} features")
    print(f"   Medium scale (10 < |value| <= 1000): {len(medium_scale)} features")
    print(f"   Small scale (|value| <= 10): {len(small_scale)} features")
    
    if len(large_scale) > 0:
        print(f"\n⚠️  Features with LARGE scale (consider normalization):")
        for i, feat in enumerate(list(large_scale['Feature']), 1):
            print(f"   {i:2d}. {feat}")
    
    if len(medium_scale) > 0:
        print(f"\n📊 Features with MEDIUM scale:")
        medium_list = list(medium_scale['Feature'])
        for i, feat in enumerate(medium_list[:10], 1):
            print(f"   {i:2d}. {feat}")
        if len(medium_list) > 10:
            print(f"   ... and {len(medium_list) - 10} more")
    
    if len(small_scale) > 0:
        print(f"\n✅ Features with SMALL scale (likely normalized):")
        small_list = list(small_scale['Feature'])
        for i, feat in enumerate(small_list[:10], 1):
            print(f"   {i:2d}. {feat}")
        if len(small_list) > 10:
            print(f"   ... and {len(small_list) - 10} more")
    
    # Additional statistics
    print(f"\n📈 Additional Statistics:")
    print(f"   Average Range: {scale_df['Range'].mean():.2f}")
    print(f"   Median Range: {scale_df['Range'].median():.2f}")
    print(f"   Max Range: {scale_df['Range'].max():.2f}")
    print(f"   Average Std: {scale_df['Std'].mean():.2f}")
    print(f"   Median Std: {scale_df['Std'].median():.2f}")

