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
returns_df = pd.read_csv('data_2/wandb_export_2026-03-08T17_19_48.716-07_00.csv')
data_usage_df = pd.read_csv('data_2/wandb_export_2026-03-08T17_20_04.549-07_00.csv')

# Load decay experiment data
decay_returns_df = pd.read_csv('data_3/returns.csv')
decay_data_usage_df = pd.read_csv('data_3/data_usage.csv')

# Define color scheme matching the description
colors = {
    'red': '#d62728',    # Naive split learning (no compression)
    'blue': '#1f77b4',   # 90th percentile
    'green': '#2ca02c',  # 99th percentile
    'orange': '#ff7f0e', # 90p + 0.99 decay
    'purple': '#9467bd', # 90p + 0.95 decay
}

# Column names (original experiment)
none_return = 'baseline - charts/episodic_return'
p90_return = '1772388591-grad-compression=accumulate-grads - charts/episodic_return'
p99_return = 'accumulate-grads_30kwarmup - charts/episodic_return'

none_data = 'baseline - charts/network_transfer_in_mb'
p90_data = '1772388591-grad-compression=accumulate-grads - charts/network_transfer_in_mb'
p99_data = 'accumulate-grads_30kwarmup - charts/network_transfer_in_mb'

# Column names (decay experiment)
decay_none_return = '1774138806-grad-compression=None - charts/episodic_return'
decay_p90_return = '1774140526-grad-compression=accumulate-grads - charts/episodic_return'
decay_095_return = '90p gc + 0.95 decay - charts/episodic_return'
decay_099_return = '90p gc + 0.99 decay - charts/episodic_return'

decay_none_data = '1774138806-grad-compression=None - charts/network_transfer_in_mb'
decay_p90_data = '1774140526-grad-compression=accumulate-grads - charts/network_transfer_in_mb'
decay_095_data = '90p gc + 0.95 decay - charts/network_transfer_in_mb'
decay_099_data = '90p gc + 0.99 decay - charts/network_transfer_in_mb'

# Apply time-weighted EMA smoothing
def ema_smooth(data, steps, alpha=0.9):
    """Apply time-weighted exponential moving average smoothing.
    The decay is adjusted by the actual time gap between observations,
    normalized by the median step interval, so alpha=0.9 corresponds
    to one median-length step of decay."""
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
    """Smooth only the rows where a run actually logged a value,
    then interpolate back onto the full step index — matching W&B behaviour."""
    mask = df[col].notna()
    sub = df.loc[mask, ['global_step', col]].copy()
    sub['smoothed'] = ema_smooth(sub[col].values, sub['global_step'].values)
    # Reindex back to all steps via linear interpolation
    result = sub.set_index('global_step')['smoothed'].reindex(df['global_step']).interpolate('index')
    return result.values

# ============================================================
# Main figures now use data_3 (latest runs, 4 configurations)
# ============================================================
MAX_STEP = 723000

# Smooth returns data
decay_returns_df['none_smooth'] = smooth_series(decay_returns_df, decay_none_return)
decay_returns_df['p90_smooth'] = smooth_series(decay_returns_df, decay_p90_return)
decay_returns_df['d095_smooth'] = smooth_series(decay_returns_df, decay_095_return)
decay_returns_df['d099_smooth'] = smooth_series(decay_returns_df, decay_099_return)

# Cut to shortest run
decay_returns_cut = decay_returns_df[decay_returns_df['global_step'] <= MAX_STEP].copy()
decay_data_usage_cut = decay_data_usage_df[decay_data_usage_df['global_step'] <= MAX_STEP].copy()

# Figure 1: Episodic Returns (all 4 configurations)
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(decay_returns_cut['global_step'], decay_returns_cut['none_smooth'],
        color=colors['red'], linewidth=1.5, label='No Compression', alpha=0.9)
ax.plot(decay_returns_cut['global_step'], decay_returns_cut['p90_smooth'],
        color=colors['blue'], linewidth=1.5, label='90th Percentile', alpha=0.9)
ax.plot(decay_returns_cut['global_step'], decay_returns_cut['d099_smooth'],
        color=colors['orange'], linewidth=1.5, label='90th Percentile + 0.99 Decay', alpha=0.9)
ax.plot(decay_returns_cut['global_step'], decay_returns_cut['d095_smooth'],
        color=colors['purple'], linewidth=1.5, label='90th Percentile + 0.95 Decay', alpha=0.9)

ax.axvline(x=30000, color='black', linestyle='--', linewidth=1,
           label='Warm Start End', alpha=0.6)

ax.set_xlabel('Training Steps', fontsize=11)
ax.set_ylabel('Episodic Return (EMA)', fontsize=11)
ax.set_title('Training Performance Comparison', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_xlim(0, MAX_STEP)

plt.tight_layout()
plt.savefig('figures/episodic_returns.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/episodic_returns.png', bbox_inches='tight', dpi=300)
print("Saved episodic_returns.pdf and episodic_returns.png")

# Figure 2: Network Transfer (all 4 configurations)
fig, ax = plt.subplots(figsize=(7, 4))

for col, label, color in [
    (decay_none_data, 'No Compression', colors['red']),
    (decay_p90_data, '90th Percentile', colors['blue']),
    (decay_099_data, '90th Percentile + 0.99 Decay', colors['orange']),
    (decay_095_data, '90th Percentile + 0.95 Decay', colors['purple']),
]:
    mask = decay_data_usage_cut[col].notna()
    ax.plot(decay_data_usage_cut.loc[mask, 'global_step'],
            decay_data_usage_cut.loc[mask, col] / 1024,
            color=color, linewidth=1.5, label=label, alpha=0.9)

ax.axvline(x=30000, color='black', linestyle='--', linewidth=1,
           label='Warm Start End', alpha=0.6)

ax.set_xlabel('Training Steps', fontsize=11)
ax.set_ylabel('Cumulative Data Transfer (GB)', fontsize=11)
ax.set_title('Communication Overhead Comparison', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_xlim(0, MAX_STEP)

plt.tight_layout()
plt.savefig('figures/network_transfer.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/network_transfer.png', bbox_inches='tight', dpi=300)
print("Saved network_transfer.pdf and network_transfer.png")

# Calculate final statistics
print("\n=== Final Statistics (at 723k steps) ===")
for name, col in [('No Compression', 'none_smooth'), ('90th Percentile', 'p90_smooth'),
                   ('90p + 0.99 Decay', 'd099_smooth'), ('90p + 0.95 Decay', 'd095_smooth')]:
    val = decay_returns_cut[col].dropna().iloc[-1]
    print(f"{name}: final EMA return = {val:.2f}")

for name, col in [('No Compression', decay_none_data), ('90th Percentile', decay_p90_data),
                   ('90p + 0.99 Decay', decay_099_data), ('90p + 0.95 Decay', decay_095_data)]:
    mask = decay_data_usage_cut[col].notna()
    if mask.any():
        val = decay_data_usage_cut.loc[mask, col].iloc[-1] / 1024
        print(f"{name}: cumulative transfer = {val:.2f} GB")

plt.close('all')

# ============================================================
# Data 4: New experiment runs (3 configurations, returns only)
# ============================================================
exp4_df = pd.read_csv('data_4/wandb_export_2026-04-24T00_31_24.510-07_00.csv')

exp4_none = '1776624639-grad-compression=none - charts/episodic_return'
exp4_p90 = '1776628684-grad-compression=accumulate-grads - charts/episodic_return'
exp4_p90_decay = '1776636519-grad-compression=accumulate-grads - charts/episodic_return'

# Cut to shortest run (~692k steps for baseline)
EXP4_MAX_STEP = 691990
exp4_cut = exp4_df[exp4_df['global_step'] <= EXP4_MAX_STEP].copy()

exp4_cut['none_smooth'] = smooth_series(exp4_cut, exp4_none)
exp4_cut['p90_smooth'] = smooth_series(exp4_cut, exp4_p90)
exp4_cut['p90_decay_smooth'] = smooth_series(exp4_cut, exp4_p90_decay)

# Figure: Episodic Returns (3 configurations)
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(exp4_cut['global_step'], exp4_cut['none_smooth'],
        color=colors['red'], linewidth=1.5, label='No Compression', alpha=0.9)
ax.plot(exp4_cut['global_step'], exp4_cut['p90_smooth'],
        color=colors['blue'], linewidth=1.5, label='90th Percentile', alpha=0.9)
ax.plot(exp4_cut['global_step'], exp4_cut['p90_decay_smooth'],
        color=colors['orange'], linewidth=1.5, label='90th Percentile + 0.99 Decay', alpha=0.9)

ax.axvline(x=30000, color='black', linestyle='--', linewidth=1,
           label='Warm Start End', alpha=0.6)

ax.set_xlabel('Training Steps', fontsize=11)
ax.set_ylabel('Episodic Return (EMA)', fontsize=11)
ax.set_title('Training Performance Comparison (Concurrent Runs)', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_xlim(0, EXP4_MAX_STEP)

plt.tight_layout()
plt.savefig('figures/exp4_episodic_returns.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/exp4_episodic_returns.png', bbox_inches='tight', dpi=300)
print("Saved exp4_episodic_returns.pdf and exp4_episodic_returns.png")

# Statistics
print("\n=== Exp4 Final Statistics ===")
for name, col in [('No Compression', 'none_smooth'), ('90th Percentile', 'p90_smooth'),
                   ('90p + 0.99 Decay', 'p90_decay_smooth')]:
    val = exp4_cut[col].dropna().iloc[-1]
    print(f"{name}: final EMA return = {val:.2f}")

plt.close('all')

# ============================================================
# Data 5: Network transfer for exp4 runs (3 configurations)
# ============================================================
net_df = pd.read_csv('data_5/network_transfer.csv')

# The Step column is the W&B step counter. Baseline ends at 2179,
# p90 at ~1962, p90_decay at ~2054. Extrapolate shorter runs to 2179.
max_step = net_df['Step'].max()

for col in ['p90', 'p90_decay']:
    # Get non-NaN rows for this column
    valid = net_df[['Step', col]].dropna()
    # Use post-warm-start data (step > 100) for linear fit
    fit_data = valid[valid['Step'] > 100]
    slope, intercept = np.polyfit(fit_data['Step'], fit_data[col], 1)

    # Find steps where this column is NaN but baseline exists
    missing_steps = net_df.loc[net_df[col].isna() & net_df['Step'].notna(), 'Step']
    # Only extrapolate for steps beyond the last valid point
    last_valid_step = valid['Step'].max()
    extrap_steps = missing_steps[missing_steps > last_valid_step]

    for s in extrap_steps:
        net_df.loc[net_df['Step'] == s, col] = slope * s + intercept

# Save extrapolated data
net_df.to_csv('data_5/network_transfer_extrapolated.csv', index=False)

# Figure: Cumulative Network Transfer (3 configurations)
fig, ax = plt.subplots(figsize=(7, 4))

for col, label, color in [
    ('baseline', 'No Compression', colors['red']),
    ('p90', '90th Percentile', colors['blue']),
    ('p90_decay', '90th Percentile + 0.99 Decay', colors['orange']),
]:
    valid = net_df[['Step', col]].dropna()
    ax.plot(valid['Step'], valid[col] / 1024,
            color=color, linewidth=1.5, label=label, alpha=0.9)

ax.axvline(x=98, color='black', linestyle='--', linewidth=1,
           label='Warm Start End', alpha=0.6)

ax.set_xlabel('Training Iterations', fontsize=11)
ax.set_ylabel('Cumulative Data Transfer (GB)', fontsize=11)
ax.set_title('Communication Overhead Comparison', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_xlim(0, max_step)

plt.tight_layout()
plt.savefig('figures/exp4_network_transfer.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/exp4_network_transfer.png', bbox_inches='tight', dpi=300)
print("Saved exp4_network_transfer.pdf and exp4_network_transfer.png")

# Statistics
print("\n=== Exp4 Network Transfer Statistics ===")
for name, col in [('No Compression', 'baseline'), ('90th Percentile', 'p90'),
                   ('90p + 0.99 Decay', 'p90_decay')]:
    valid = net_df[col].dropna()
    val = valid.iloc[-1] / 1024
    print(f"{name}: cumulative transfer = {val:.2f} GB")

plt.close('all')

# ============================================================
# Data 6: Five configurations with different seed
# ============================================================
exp6_df = pd.read_csv('data_6/wandb_export_2026-05-04T07_48_24.081-07_00.csv')

exp6_cols = {
    'baseline': '1776981725-grad-compression=none - charts/episodic_return',
    'p90': '90p - charts/episodic_return',
    'p90_decay': '1777001996-grad-compression=accumulate-grads - charts/episodic_return',
    'p95_decay': '1777181264-grad-compression=accumulate-grads - charts/episodic_return',
    'p99_decay': '1777181322-grad-compression=accumulate-grads - charts/episodic_return',
}

# Find the shortest run (last non-NaN global_step per column)
max_steps = []
for col in exp6_cols.values():
    last_valid = exp6_df.loc[exp6_df[col].notna(), 'global_step'].max()
    max_steps.append(last_valid)
EXP6_MAX_STEP = min(max_steps)
print(f"\nExp6: clipping to {EXP6_MAX_STEP} steps (shortest run)")

exp6_cut = exp6_df[exp6_df['global_step'] <= EXP6_MAX_STEP].copy()

# Smooth each series
for key, col in exp6_cols.items():
    exp6_cut[f'{key}_smooth'] = smooth_series(exp6_cut, col)

# Figure: Episodic Returns (5 configurations)
fig, ax = plt.subplots(figsize=(7, 4))

plot_configs = [
    ('baseline_smooth', 'No Compression', colors['red']),
    ('p90_smooth', '90th Percentile', colors['blue']),
    ('p90_decay_smooth', '90th Percentile + 0.99 Decay', colors['orange']),
    ('p95_decay_smooth', '95th Percentile + 0.99 Decay', colors['green']),
    ('p99_decay_smooth', '99th Percentile + 0.99 Decay', colors['purple']),
]

for col, label, color in plot_configs:
    ax.plot(exp6_cut['global_step'], exp6_cut[col],
            color=color, linewidth=1.5, label=label, alpha=0.9)

ax.axvline(x=30000, color='black', linestyle='--', linewidth=1,
           label='Warm Start End', alpha=0.6)

ax.set_xlabel('Training Steps', fontsize=11)
ax.set_ylabel('Episodic Return (EMA)', fontsize=11)
ax.set_title('Training Performance Across Percentile Thresholds', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=8)
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_xlim(0, EXP6_MAX_STEP)

plt.tight_layout()
plt.savefig('figures/exp6_episodic_returns.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/exp6_episodic_returns.png', bbox_inches='tight', dpi=300)
print("Saved exp6_episodic_returns.pdf and exp6_episodic_returns.png")

# Statistics
print("\n=== Exp6 Final Statistics ===")
for key, col in exp6_cols.items():
    val = exp6_cut[f'{key}_smooth'].dropna().iloc[-1]
    print(f"{key}: final EMA return = {val:.2f}")

plt.close('all')
