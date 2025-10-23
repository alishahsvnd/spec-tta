#!/usr/bin/env python3
"""
Visualize PETSA vs SPEC-TTA comparison on seasonal shift data.
"""

import matplotlib.pyplot as plt
import numpy as np

# Results
methods = ['No Adaptation', 'PETSA\n(rank-4)', 'SPEC-TTA\n(K=32)']
mse_values = [0.5417, 0.5399, 0.1857]
mae_values = [0.3566, 0.3543, 0.3400]
params = [0, 7502, 910]
updates = [0, 143, 30]

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('PETSA vs SPEC-TTA on Seasonal Shift Data (ETTh1)', fontsize=16, fontweight='bold')

# 1. MSE Comparison
ax1 = axes[0, 0]
bars1 = ax1.bar(methods, mse_values, color=['gray', '#ff7f0e', '#2ca02c'], alpha=0.8, edgecolor='black')
ax1.set_ylabel('Test MSE', fontsize=12, fontweight='bold')
ax1.set_title('Test MSE (Lower is Better)', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar, val in zip(bars1, mse_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.4f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add improvement annotation
improvement = (mse_values[1] - mse_values[2]) / mse_values[1] * 100
ax1.annotate(f'65.6% Improvement\nover PETSA!',
             xy=(2, mse_values[2]), xytext=(1.5, 0.4),
             arrowprops=dict(arrowstyle='->', color='green', lw=2),
             fontsize=11, color='green', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

# 2. MAE Comparison
ax2 = axes[0, 1]
bars2 = ax2.bar(methods, mae_values, color=['gray', '#ff7f0e', '#2ca02c'], alpha=0.8, edgecolor='black')
ax2.set_ylabel('Test MAE', fontsize=12, fontweight='bold')
ax2.set_title('Test MAE (Lower is Better)', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

for bar, val in zip(bars2, mae_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.4f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 3. Parameter Efficiency
ax3 = axes[1, 0]
bars3 = ax3.bar(['PETSA', 'SPEC-TTA'], [params[1], params[2]], 
                color=['#ff7f0e', '#2ca02c'], alpha=0.8, edgecolor='black')
ax3.set_ylabel('Trainable Parameters', fontsize=12, fontweight='bold')
ax3.set_title('Parameter Efficiency', fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3, linestyle='--')

for bar, val in zip(bars3, [params[1], params[2]]):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:,}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax3.annotate('8.2x Fewer\nParameters!',
             xy=(1, params[2]), xytext=(0.5, 5000),
             arrowprops=dict(arrowstyle='->', color='green', lw=2),
             fontsize=11, color='green', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

# 4. Adaptation Updates
ax4 = axes[1, 1]
bars4 = ax4.bar(['PETSA', 'SPEC-TTA'], [updates[1], updates[2]], 
                color=['#ff7f0e', '#2ca02c'], alpha=0.8, edgecolor='black')
ax4.set_ylabel('Number of Adaptation Updates', fontsize=12, fontweight='bold')
ax4.set_title('Adaptation Selectivity', fontsize=13, fontweight='bold')
ax4.grid(axis='y', alpha=0.3, linestyle='--')

for bar, val in zip(bars4, [updates[1], updates[2]]):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{val}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax4.annotate('4.7x More\nSelective!',
             xy=(1, updates[2]), xytext=(0.5, 80),
             arrowprops=dict(arrowstyle='->', color='green', lw=2),
             fontsize=11, color='green', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig('/home/alishah/PETSA/petsa_vs_spec_tta_fixed.png', dpi=300, bbox_inches='tight')
plt.savefig('/home/alishah/PETSA/petsa_vs_spec_tta_fixed.pdf', bbox_inches='tight')
print(f"✅ Saved comparison plots to petsa_vs_spec_tta_fixed.png/pdf")

# Print summary table
print("\n" + "="*70)
print("SPEC-TTA vs PETSA Comparison on Seasonal Shift Data")
print("="*70)
print(f"{'Method':<20} {'MSE':<12} {'MAE':<12} {'Params':<12} {'Updates':<10}")
print("-"*70)
print(f"{'No Adaptation':<20} {mse_values[0]:<12.4f} {mae_values[0]:<12.4f} {params[0]:<12} {updates[0]:<10}")
print(f"{'PETSA (rank-4)':<20} {mse_values[1]:<12.4f} {mae_values[1]:<12.4f} {params[1]:<12} {updates[1]:<10}")
print(f"{'SPEC-TTA (K=32)':<20} {mse_values[2]:<12.4f} {mae_values[2]:<12.4f} {params[2]:<12} {updates[2]:<10}")
print("="*70)
print(f"\n🎯 SPEC-TTA Improvements:")
print(f"   • MSE Reduction: {(mse_values[1] - mse_values[2]) / mse_values[1] * 100:.1f}% better than PETSA")
print(f"   • Parameter Efficiency: {params[1] / params[2]:.1f}x fewer parameters")
print(f"   • Adaptation Selectivity: {updates[1] / updates[2]:.1f}x more selective")
print(f"   • MAE Improvement: {(mae_values[1] - mae_values[2]) / mae_values[1] * 100:.1f}% better than PETSA")
print("="*70)
