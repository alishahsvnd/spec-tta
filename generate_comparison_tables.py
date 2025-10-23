#!/usr/bin/env python3
"""
Generate comparison tables for SPEC-TTA vs PETSA paper
"""

import pandas as pd

# Data from experiments
data = {
    'Model': ['iTransformer', 'DLinear', 'PatchTST'] * 4,
    'Horizon': [96]*3 + [192]*3 + [336]*3 + [720]*3,
    'Baseline': [
        0.2656, 0.2323, 0.2383,  # H=96
        0.2996, 0.2862, 0.2804,  # H=192
        0.3548, 0.3252, 0.3201,  # H=336
        0.4298, 0.4087, 0.4268,  # H=720
    ],
    'PETSA': [
        0.2648, 0.2313, 0.2382,  # H=96
        0.2932, 0.2837, 0.2764,  # H=192
        0.3366, 0.3199, 0.3117,  # H=336
        0.4086, 0.3910, 0.4000,  # H=720
    ],
    'SPEC-TTA': [
        0.2279, 0.1529, 0.1559,  # H=96
        0.2404, 0.1971, 0.1656,  # H=192
        1.3064, 0.4051, 0.3433,  # H=336
        1.3324, 0.4086, 0.4268,  # H=720
    ],
}

df = pd.DataFrame(data)

# Calculate improvements
df['PETSA_Improv_%'] = ((df['Baseline'] - df['PETSA']) / df['Baseline'] * 100).round(1)
df['SPECTTA_Improv_%'] = ((df['Baseline'] - df['SPEC-TTA']) / df['Baseline'] * 100).round(1)
df['Winner'] = df.apply(lambda x: 'SPEC-TTA' if x['SPECTTA_Improv_%'] > x['PETSA_Improv_%'] else 'PETSA', axis=1)
df['Margin_%'] = (df['SPECTTA_Improv_%'] - df['PETSA_Improv_%']).round(1)

# Print comprehensive table
print("="*100)
print("COMPLETE COMPARISON: SPEC-TTA vs PETSA on ETTh2")
print("="*100)
print()

for horizon in [96, 192, 336, 720]:
    subset = df[df['Horizon'] == horizon]
    print(f"\n{'='*100}")
    print(f"HORIZON {horizon}")
    print(f"{'='*100}")
    print(subset[['Model', 'Baseline', 'PETSA', 'PETSA_Improv_%', 'SPEC-TTA', 'SPECTTA_Improv_%', 'Winner', 'Margin_%']].to_string(index=False))
    
    avg_petsa = subset['PETSA_Improv_%'].mean()
    avg_spec = subset['SPECTTA_Improv_%'].mean()
    
    print(f"\nAverage Improvement:")
    print(f"  PETSA:    {avg_petsa:+6.1f}%")
    print(f"  SPEC-TTA: {avg_spec:+6.1f}%")
    print(f"  Winner:   {'SPEC-TTA' if avg_spec > avg_petsa else 'PETSA'} by {abs(avg_spec - avg_petsa):.1f}%")

# Overall summary
print(f"\n{'='*100}")
print("OVERALL SUMMARY")
print(f"{'='*100}")

summary_data = []
for horizon in [96, 192, 336, 720]:
    subset = df[df['Horizon'] == horizon]
    summary_data.append({
        'Horizon': horizon,
        'PETSA_Avg_%': subset['PETSA_Improv_%'].mean().round(1),
        'SPECTTA_Avg_%': subset['SPECTTA_Improv_%'].mean().round(1),
        'Winner': 'SPEC-TTA' if subset['SPECTTA_Improv_%'].mean() > subset['PETSA_Improv_%'].mean() else 'PETSA'
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# LaTeX table for paper
print(f"\n{'='*100}")
print("LATEX TABLE FOR PAPER")
print(f"{'='*100}")
print("""
\\begin{table*}[t]
\\centering
\\caption{Comparison of SPEC-TTA and PETSA on ETTh2 across multiple horizons. Values show MSE reduction percentage (higher is better). Bold indicates winner for each horizon.}
\\label{tab:spec_petsa_comparison}
\\begin{tabular}{lcccccccc}
\\toprule
& \\multicolumn{2}{c}{H=96} & \\multicolumn{2}{c}{H=192} & \\multicolumn{2}{c}{H=336} & \\multicolumn{2}{c}{H=720} \\\\
\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}
Model & PETSA & SPEC-TTA & PETSA & SPEC-TTA & PETSA & SPEC-TTA & PETSA & SPEC-TTA \\\\
\\midrule
""")

for model in ['iTransformer', 'DLinear', 'PatchTST']:
    model_data = df[df['Model'] == model]
    line = f"{model:15}"
    for horizon in [96, 192, 336, 720]:
        row = model_data[model_data['Horizon'] == horizon].iloc[0]
        petsa = row['PETSA_Improv_%']
        spec = row['SPECTTA_Improv_%']
        
        if spec > petsa:
            line += f" & {petsa:+5.1f}\\% & \\textbf{{{spec:+5.1f}\\%}}"
        else:
            line += f" & \\textbf{{{petsa:+5.1f}\\%}} & {spec:+5.1f}\\%"
    line += " \\\\"
    print(line)

print("\\midrule")
avg_line = "Average        "
for horizon in [96, 192, 336, 720]:
    subset = df[df['Horizon'] == horizon]
    petsa_avg = subset['PETSA_Improv_%'].mean()
    spec_avg = subset['SPECTTA_Improv_%'].mean()
    
    if spec_avg > petsa_avg:
        avg_line += f" & {petsa_avg:+5.1f}\\% & \\textbf{{{spec_avg:+5.1f}\\%}}"
    else:
        avg_line += f" & \\textbf{{{petsa_avg:+5.1f}\\%}} & {spec_avg:+5.1f}\\%"
avg_line += " \\\\"
print(avg_line)

print("""\\bottomrule
\\end{tabular}
\\end{table*}
""")

print(f"\n{'='*100}")
print("KEY FINDINGS")
print(f"{'='*100}")
print(f"1. SPEC-TTA dominates on short horizons (H≤192): +29.2% vs +0.9% average")
print(f"2. PETSA dominates on long horizons (H≥336): +4.2% vs -85.0% average")
print(f"3. SPEC-TTA has catastrophic failures on H≥336 (iTransformer: -268%)")
print(f"4. PETSA is stable but marginal on short horizons")
print(f"5. Recommendation: Use SPEC-TTA for H≤192, PETSA for H≥336")
print(f"{'='*100}")
