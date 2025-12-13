import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (7, 4)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

# Load data
returns_df = pd.read_csv('data/returns.csv')
data_usage_df = pd.read_csv('data/data_usage.csv')

# Define color scheme matching the description
colors = {
    'red': '#d62728',    # Naive split learning (no compression)
    'blue': '#1f77b4',   # 90th percentile
    'green': '#2ca02c'   # 99th percentile
}

# Column names
none_return = '1765346213-grad-compression=None - charts/episodic_return'
p90_return = '1765346490-grad-compression=accumulate-grads - charts/episodic_return'
p99_return = '1765346631-grad-compression=accumulate-grads - charts/episodic_return'

none_data = '1765346213-grad-compression=None - charts/network_transfer_in_mb'
p90_data = '1765346490-grad-compression=accumulate-grads - charts/network_transfer_in_mb'
p99_data = '1765346631-grad-compression=accumulate-grads - charts/network_transfer_in_mb'

# Apply EMA smoothing
def ema_smooth(data, alpha=0.9):
    """Apply exponential moving average smoothing"""
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        if not np.isnan(data[i]):
            smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * data[i]
        else:
            smoothed[i] = smoothed[i-1]
    return smoothed

# Smooth returns data
returns_df['none_smooth'] = ema_smooth(returns_df[none_return].fillna(method='ffill'))
returns_df['p90_smooth'] = ema_smooth(returns_df[p90_return].fillna(method='ffill'))
returns_df['p99_smooth'] = ema_smooth(returns_df[p99_return].fillna(method='ffill'))

# Figure 1: Episodic Returns
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(returns_df['global_step'], returns_df['none_smooth'], 
        color=colors['red'], linewidth=1.5, label='No Compression', alpha=0.9)
ax.plot(returns_df['global_step'], returns_df['p90_smooth'], 
        color=colors['blue'], linewidth=1.5, label='90th Percentile', alpha=0.9)
ax.plot(returns_df['global_step'], returns_df['p99_smooth'], 
        color=colors['green'], linewidth=1.5, label='99th Percentile', alpha=0.9)

# Add warm start line
ax.axvline(x=30000, color='black', linestyle='--', linewidth=1, 
           label='Warm Start End', alpha=0.6)

ax.set_xlabel('Training Steps', fontsize=11)
ax.set_ylabel('Episodic Return (EMA)', fontsize=11)
ax.set_title('Training Performance Comparison', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_xlim(0, returns_df['global_step'].max())

plt.tight_layout()
plt.savefig('figures/episodic_returns.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/episodic_returns.png', bbox_inches='tight', dpi=300)
print("Saved episodic_returns.pdf and episodic_returns.png")

# Figure 2: Network Transfer
fig, ax = plt.subplots(figsize=(7, 4))

# Filter out NaN values and plot
none_mask = ~data_usage_df[none_data].isna()
p90_mask = ~data_usage_df[p90_data].isna()
p99_mask = ~data_usage_df[p99_data].isna()

ax.plot(data_usage_df.loc[none_mask, 'global_step'], 
        data_usage_df.loc[none_mask, none_data] / 1024,  # Convert to GB
        color=colors['red'], linewidth=1.5, label='No Compression', alpha=0.9)
ax.plot(data_usage_df.loc[p90_mask, 'global_step'], 
        data_usage_df.loc[p90_mask, p90_data] / 1024,
        color=colors['blue'], linewidth=1.5, label='90th Percentile', alpha=0.9)
ax.plot(data_usage_df.loc[p99_mask, 'global_step'], 
        data_usage_df.loc[p99_mask, p99_data] / 1024,
        color=colors['green'], linewidth=1.5, label='99th Percentile', alpha=0.9)

# Add warm start line
ax.axvline(x=30000, color='black', linestyle='--', linewidth=1, 
           label='Warm Start End', alpha=0.6)

ax.set_xlabel('Training Steps', fontsize=11)
ax.set_ylabel('Cumulative Data Transfer (GB)', fontsize=11)
ax.set_title('Communication Overhead Comparison', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_xlim(0, data_usage_df['global_step'].max())

plt.tight_layout()
plt.savefig('figures/network_transfer.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/network_transfer.png', bbox_inches='tight', dpi=300)
print("Saved network_transfer.pdf and network_transfer.png")

# Calculate final statistics
final_step_idx = returns_df['global_step'].idxmax()
final_none = data_usage_df.loc[none_mask, none_data].iloc[-1] / 1024
final_p90 = data_usage_df.loc[p90_mask, p90_data].iloc[-1] / 1024
final_p99 = data_usage_df.loc[p99_mask, p99_data].iloc[-1] / 1024

print("\n=== Final Statistics ===")
print(f"No Compression: {final_none:.2f} GB")
print(f"90th Percentile: {final_p90:.2f} GB ({(1 - final_p90/final_none)*100:.1f}% reduction)")
print(f"99th Percentile: {final_p99:.2f} GB ({(1 - final_p99/final_none)*100:.1f}% reduction)")

# Calculate final returns
final_return_none = returns_df['none_smooth'].iloc[-1]
final_return_p90 = returns_df['p90_smooth'].iloc[-1]
final_return_p99 = returns_df['p99_smooth'].iloc[-1]

print(f"\nFinal Returns (EMA):")
print(f"No Compression: {final_return_none:.2f}")
print(f"90th Percentile: {final_return_p90:.2f}")
print(f"99th Percentile: {final_return_p99:.2f}")

plt.close('all')
