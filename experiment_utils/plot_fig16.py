import argparse

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from exp_name_utils import augment_df

parser = argparse.ArgumentParser()
parser.add_argument('--ablation_data', nargs="+", required=True,
                    help="Paths to the jsonl file containing the ablation "
                    "data (grid), generated by collect_throughput_stats.py")
parser.add_argument('--out_dir', type=str, required=True,
                    help="Path to the directory where the output plots "
                    "will be saved")

args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

dfs = []
for path in args.ablation_data:
    df = pd.read_json(path, lines=True)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df = augment_df(df)

def calculate_throughput(row):
    num_tokens = row["num_tokens"]
    num_iters = row["num_iters"]
    total_time = row["avg_iter_time"] * num_iters # time is in ms
    return (num_tokens / total_time) * 1000

df["throughput"] = df.apply(calculate_throughput, axis=1)

def parse_exp_type(row):
    if "sch" in row["spec_name"]:
        return "Schedule"
    else:
        return "Batching"

df["Exp Type"] = df.apply(parse_exp_type, axis=1)

def parse_batching_ablation_type(row):
    if "token_based" in row["spec_name"] or "notsp" in row["spec_name"]:
        name = ""
        if "token_based" in row["spec_name"]:
            name += "TB"
        else:
            name += "DP"
        if "notsp" in row["spec_name"]:
            name += " (S)"
        else:
            name += " (T)"
    else:
        if row["framework"] == "baseline":
            name = "MLM\n+DS"
        elif row["framework"] == "dynapipe":
            name = "DP (T)"
        else:
            assert False, f"Unknown framework: {row['framework']}"
    return name

def parse_schedule_ablation_type(row):
    if "wait-free-cyclic_noperm" in row["spec_name"]:
        return "Adaptive\n(no reorder)"
    elif "wait-free-cyclic" in row["spec_name"]:
        return "Adaptive"
    else:
        return "1F1B"

batching_df = df[df["Exp Type"] == "Batching"].copy()
scheduling_df = df[df["Exp Type"] == "Schedule"].copy()

batching_df["Type"] = batching_df.apply(parse_batching_ablation_type, axis=1)
scheduling_df["Type"] = scheduling_df.apply(parse_schedule_ablation_type, axis=1)

# preprocessing for fig16a
# select the best result from grid search
def is_grid(row):
    return "grid" in row["exp_name"]

batching_df["IsGrid"] = batching_df.apply(is_grid, axis=1)
batching_grid_df = batching_df[batching_df["IsGrid"] == True]
batching_non_grid_df = batching_df[batching_df["IsGrid"] == False]

batching_grid_df = batching_grid_df.loc[batching_grid_df.groupby(["Type"])["throughput"].idxmax()]

batching_df = pd.concat([batching_grid_df, batching_non_grid_df], ignore_index=True)

batching_df["Type"] = pd.Categorical(batching_df["Type"], ["MLM\n+DS", "TB (S)", "TB (T)", "DP (S)", "DP (T)"])

# preprocessing for fig16b
scheduling_df["Type"] = pd.Categorical(scheduling_df["Type"], ["1F1B", "Adaptive\n(no reorder)", "Adaptive"], ordered=True)
scheduling_df["Global Batch Size"] = scheduling_df["global_batch_size"]
# transform throughput -> speedup (normalized by the slowest)
scheduling_df["throughput"] = scheduling_df.groupby('Global Batch Size')["throughput"].transform(lambda x: (x / x.min()))

# fig16a
fig, ax = plt.subplots(figsize=(4, 4))

sns.barplot(data=batching_df, x="Type", y="throughput", hue="Type", ax=ax, orient='v')
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.set_ylabel("Throughput (tokens/s)", fontsize=16)
ax.set_xlabel("Micro-batching Method", fontsize=16)
fig.savefig(os.path.join(args.out_dir, "fig16_a.pdf"), bbox_inches="tight")

# fig16b
fig, ax = plt.subplots(figsize=(4, 3))
ax = sns.pointplot(data=scheduling_df, x="Type", y="throughput", hue="Global Batch Size",
                   ax=ax, palette=sns.color_palette(n_colors=2))
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(14)
ax.xaxis.set_tick_params(labelsize=12)
ax.set_ylabel("Normalized Throughput", fontsize=14)
ax.set_xlabel("Schedule Method", fontsize=14)
ax.legend(loc="lower right", title="Global Batch Size")
plt.setp(ax.get_legend().get_texts(), fontsize='10') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='10') # for legend title
fig.savefig(os.path.join(args.out_dir, "fig16_b.pdf"), bbox_inches="tight")





