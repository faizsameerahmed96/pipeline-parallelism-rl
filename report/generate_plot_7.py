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

colors = {
    'red': '#d62728',
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
}

# Apply time-weighted EMA smoothing
def ema_smooth(data, steps, alpha=0.9):
    data = np.array(data, dtype=float)
    steps = np.array(steps, dtype=float)
    dt = np.diff(steps)
    dt_ref = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0

    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        delta = steps[i] - steps[i - 1]
        alpha_eff = alpha ** (delta / dt_ref)
        if not np.isnan(data[i]):
            smoothed[i] = alpha_eff * smoothed[i - 1] + (1 - alpha_eff) * data[i]
        else:
            smoothed[i] = smoothed[i - 1]
    return smoothed

def smooth_series(df, col):
    mask = df[col].notna()
    sub = df.loc[mask, ['global_step', col]].copy()
    sub['smoothed'] = ema_smooth(sub[col].values, sub['global_step'].values)
    result = sub.set_index('global_step')['smoothed'].reindex(df['global_step']).interpolate('index')
    return result.values

# Load raw data (no W&B EMA applied)
df = pd.read_csv('data_8/wandb_export_2026-05-17T13_25_27.563-07_00.csv')

# Column names
baseline_col = '1778988573-grad-compression=none - charts/episodic_return'
seed1_col = '1778997949-grad-compression=accumulate-grads - charts/episodic_return'
seed2_col = '1779039473-grad-compression=accumulate-grads - charts/episodic_return'

# Cut to 3M steps
MAX_STEP = 3_000_000
df_cut = df[df['global_step'] <= MAX_STEP].copy()

# Plot each series: drop NaNs, apply same EMA smoothing to all
fig, ax = plt.subplots(figsize=(7, 4))

WINDOW = 20

for col, color, label in [
    (baseline_col, colors['red'], 'Baseline (No Compression)'),
    (seed1_col, colors['blue'], '90p + 0.99 Decay'),
    (seed2_col, colors['orange'], '90p + 0.98 Decay'),
]:
    mask = df_cut[col].notna()
    sub = df_cut.loc[mask].copy()
    smoothed = sub[col].rolling(window=WINDOW, min_periods=1, center=True).mean()
    ax.plot(sub['global_step'].values, smoothed.values,
            color=color, linewidth=1.5, label=label, alpha=0.9)

ax.set_xlabel('Environment Steps (Millions)')
ax.set_ylabel('Episodic Return')
ax.set_title('Enduro — Gradient Compression vs Baseline')
ax.legend(loc='upper left', framealpha=0.9)
ax.set_xlim(0, MAX_STEP)
ax.set_xticks([0, 0.5e6, 1e6, 1.5e6, 2e6, 2.5e6, 3e6])
ax.set_xticklabels(['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0'])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig7_enduro_compression_3M.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig7_enduro_compression_3M.pdf', bbox_inches='tight')
print("Saved figures/fig7_enduro_compression_3M.png and .pdf")

# ============================================================
# Figure 2: Cumulative Data Transfer
# ============================================================
bw_df = pd.read_csv('data_8/wandb_export_2026-05-17T13_33_40.600-07_00.csv')

baseline_bw = '1778988573-grad-compression=none - charts/calculated_data_transfer_total_mb'
seed1_bw = '1778997949-grad-compression=accumulate-grads - charts/calculated_data_transfer_total_mb'
seed2_bw = '1779039473-grad-compression=accumulate-grads - charts/calculated_data_transfer_total_mb'

fig2, ax2 = plt.subplots(figsize=(7, 4))

MAX_ITER_3M = 3_000_000 // 1024  # ~2929 iterations = 3M steps

for col, color, label in [
    (baseline_bw, colors['red'], 'Baseline (No Compression)'),
    (seed1_bw, colors['blue'], '90p + 0.99 Decay'),
    (seed2_bw, colors['orange'], '90p + 0.98 Decay'),
]:
    mask = bw_df[col].notna()
    sub = bw_df.loc[mask].reset_index(drop=True)
    sub['global_step'] = sub.index * 1024
    sub = sub[sub['global_step'] <= MAX_STEP]
    # Convert MB to GB
    ax2.plot(sub['global_step'].values, sub[col].values / 1024,
             color=color, linewidth=1.5, label=label, alpha=0.9)

ax2.set_xlabel('Environment Steps (Millions)')
ax2.set_ylabel('Cumulative Data Transfer (GB)')
ax2.set_title('Enduro — Cumulative Data Transfer')
ax2.legend(loc='upper left', framealpha=0.9)
ax2.set_xlim(0, MAX_STEP)
ax2.set_xticks([0, 0.5e6, 1e6, 1.5e6, 2e6, 2.5e6, 3e6])
ax2.set_xticklabels(['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0'])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig8_enduro_bandwidth_3M.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig8_enduro_bandwidth_3M.pdf', bbox_inches='tight')
print("Saved figures/fig8_enduro_bandwidth_3M.png and .pdf")
