#!/usr/bin/env python3
"""
Create a visual summary plot comparing SPEC-TTA across all backbones.
"""

import matplotlib.pyplot as plt
import numpy as np

# Results data
models = ['PatchTST', 'DLinear', 'iTransformer', 'FreTS', 'MICN']
baseline_mse = [0.5396, 0.4778, 0.5417, 0.6519, 0.5665]
spec_tta_mse = [0.1173, 0.1359, 0.1857, 0.2589, 0.3398]
improvements = [78.3, 71.6, 65.7, 60.3, 40.0]

# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('SPEC-TTA Full Backbone Comparison (ETTh1, Horizon=96)', 
             fontsize=16, fontweight='bold')

# Plot 1: MSE Comparison
ax1 = axes[0]
x = np.arange(len(models))
width = 0.35

bars1 = ax1.bar(x - width/2, baseline_mse, width, label='Baseline', color='#ff7f0e', alpha=0.8)
bars2 = ax1.bar(x + width/2, spec_tta_mse, width, label='SPEC-TTA', color='#2ca02c', alpha=0.8)

ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_ylabel('MSE', fontsize=12, fontweight='bold')
ax1.set_title('MSE: Baseline vs SPEC-TTA', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=15, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Improvement Percentage
ax2 = axes[1]
colors = plt.cm.RdYlGn(np.array(improvements) / 100)
bars = ax2.barh(models, improvements, color=colors, alpha=0.8)

ax2.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
ax2.set_title('MSE Improvement by Model', fontsize=13, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, improvements)):
    ax2.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

# Add average line
avg_improvement = np.mean(improvements)
ax2.axvline(avg_improvement, color='red', linestyle='--', linewidth=2, 
           label=f'Average: {avg_improvement:.1f}%')
ax2.legend()

# Plot 3: Parameter Efficiency (Params/MSE ratio)
ax3 = axes[2]
params = 910  # Same for all models
efficiency = [params / mse for mse in spec_tta_mse]

bars = ax3.bar(models, efficiency, color='#1f77b4', alpha=0.8)
ax3.set_ylabel('Parameters / MSE', fontsize=12, fontweight='bold')
ax3.set_title('Parameter Efficiency\n(Higher = More Efficient)', fontsize=13, fontweight='bold')
ax3.set_xticklabels(models, rotation=15, ha='right')
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, efficiency):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(val)}', ha='center', va='bottom', fontsize=9)

# Add text box with summary stats
textstr = f'910 parameters per model\n30 updates per model\nAvg improvement: {avg_improvement:.1f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax3.text(0.95, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('spec_tta_comparison_summary.png', dpi=300, bbox_inches='tight')
plt.savefig('spec_tta_comparison_summary.pdf', bbox_inches='tight')
print("✓ Saved visualization to spec_tta_comparison_summary.png/pdf")

# Create a second figure: Before/After comparison
fig2, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(models))
width = 0.35

# Create grouped bars
bars1 = ax.bar(x - width/2, baseline_mse, width, label='Baseline (No TTA)', 
              color='#d62728', alpha=0.7)
bars2 = ax.bar(x + width/2, spec_tta_mse, width, label='SPEC-TTA', 
              color='#2ca02c', alpha=0.7)

# Customize
ax.set_xlabel('Time-Series Forecasting Backbone', fontsize=13, fontweight='bold')
ax.set_ylabel('Mean Squared Error (MSE)', fontsize=13, fontweight='bold')
ax.set_title('SPEC-TTA Performance Across Diverse Architectures\nDataset: ETTh1, Horizon: 96', 
            fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add improvement labels
for i, (b1, b2, imp) in enumerate(zip(baseline_mse, spec_tta_mse, improvements)):
    # Arrow showing improvement
    ax.annotate('', xy=(i, b2), xytext=(i, b1),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    # Improvement percentage
    mid_y = (b1 + b2) / 2
    ax.text(i + 0.15, mid_y, f'↓{imp:.1f}%', fontsize=9, fontweight='bold',
           color='darkgreen', rotation=0)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.4f}', ha='center', va='bottom', fontsize=8)

# Add summary text box
summary = (f"Key Results:\n"
          f"• Best MSE: 0.1173 (PatchTST, 78.3% improvement)\n"
          f"• Avg Improvement: 63.2%\n"
          f"• Parameters: 910 (same for all models)\n"
          f"• Updates: 30 (drift-triggered)")
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=10,
       verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('spec_tta_before_after.png', dpi=300, bbox_inches='tight')
plt.savefig('spec_tta_before_after.pdf', bbox_inches='tight')
print("✓ Saved before/after comparison to spec_tta_before_after.png/pdf")

print("\n" + "="*60)
print("All visualizations created successfully!")
print("="*60)
print("\nGenerated files:")
print("  1. spec_tta_comparison_summary.png/pdf - 3-panel analysis")
print("  2. spec_tta_before_after.png/pdf - Before/after comparison")
print("\nThese figures are ready for inclusion in your paper.")
