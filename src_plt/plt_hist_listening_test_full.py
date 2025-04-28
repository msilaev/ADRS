import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the Excel file

import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from statsmodels.sandbox.stats.multicomp import multipletests

fontsize_ticks =25
fontsize_labels = 30
# Load the data (update filename if needed)
file_path = "../data/listening_test_source/results_Tuni_BWE_N16_final2.xlsx"
df = pd.read_excel(file_path).dropna()

df["Condition"] = df["Condition"].apply(lambda x: " ".join(x.split(" ")[1:]))

df["Condition"] = df["Condition"].apply(lambda x: "WB" if x == "Direct" else x)
df["Condition"] = df["Condition"].apply(lambda x: "AudioUnet" if x == "Audiounet" else x)

df["Condition"] = df["Condition"].apply(lambda x: "HiFi-GAN" if x == "HifiGAN" else x)
df["Condition"] = df["Condition"].apply(lambda x: "MU-GAN" if x == "GAN" else x)

df["Condition"] = df["Condition"].apply(lambda x: "Spline 7kHz" if x == "Spline-up 7kHz" else x)

# Compute summary statistics
summary_stats = df.groupby("Condition")["Score"].agg(
    ["mean", "median", "std", "count", "min", "max"]
).reset_index()

# Compute Q1 (25th percentile) and Q3 (75th percentile)
summary_stats["Q1"] = df.groupby("Condition")["Score"].quantile(0.25).values
summary_stats["Q3"] = df.groupby("Condition")["Score"].quantile(0.75).values

# Compute Standard Error (SE) and 95% Confidence Interval (CI)
summary_stats["sem"] = summary_stats["std"] / np.sqrt(summary_stats["count"])
summary_stats["ci_95"] = summary_stats["sem"] * 1.96  # 95% CI

# Print summary statistics
print(summary_stats)

# Perform ANOVA test (two-tailed by default)
anova_result = stats.f_oneway(*[df[df["Condition"] == cond]["Score"] for cond in df["Condition"].unique()])
print(f"ANOVA p-value: {anova_result.pvalue}")

# Check if the result is significant
alpha = 0.05
if anova_result.pvalue < alpha:
    print("ANOVA: Significant differences found (p < 0.05)")
else:
    print("ANOVA: No significant differences found (p >= 0.05)")

# Perform Tukey’s HSD test for post-hoc analysis
tukey = pairwise_tukeyhsd(df["Score"], df["Condition"], alpha=0.05)
print(tukey)

# Adjusted significance levels for multiple comparisons
comparison = MultiComparison(df["Score"], df["Condition"])
tukey_result = comparison.tukeyhsd(alpha=0.05)

# Extract significance labels for visualization
group_labels = {condition: "" for condition in df["Condition"].unique()}
num_groups = len(tukey_result.groupsunique)

# Iterate through group pairs and extract p-values
for i in range(num_groups):
    for j in range(i + 1, num_groups):
        cond1, cond2 = tukey_result.groupsunique[i], tukey_result.groupsunique[j]
        index = i * (num_groups - i - 1) // 2 + (j - i - 1)  # Corrected indexing
        p_value = tukey_result.pvalues[index]
        if p_value < alpha:
            group_labels[cond1] += "*"
            group_labels[cond2] += "*"

###########################################333
# Define desired order of conditions
condition_order = summary_stats["Condition"].tolist()
print(condition_order)
condition_order = ['WB', 'AudioUnet', 'HiFi-GAN', 'LP 3.5kHz', 'MU-GAN', 'Spline 7kHz']

# Apply the same order to the main dataframe used for the boxplot
df["Condition"] = pd.Categorical(df["Condition"], categories=condition_order, ordered=True)
summary_stats["Condition"] = pd.Categorical(summary_stats["Condition"], categories=condition_order, ordered=True)

#############################################
# Convert labels to DataFrame for plotting
summary_stats["significance"] = summary_stats["Condition"].map(group_labels)

# Set figure size
plt.figure(figsize=(14, 6))

# Boxplot showing IQRs, medians, and outliers
plt.subplot(1, 2, 1)
sns.boxplot(x="Condition", y="Score", data=df, showmeans=True)
# Increase font sizes
plt.ylabel("Score", fontsize=fontsize_labels)
plt.xlabel("", fontsize=fontsize_labels)
#plt.title("MUSHRA Score Distribution (IQR & Median)", fontsize=20)
plt.xticks(rotation=45, fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.grid()

# Barplot with means, 95% CI, and significance annotations
plt.subplot(1, 2, 2)

ax = sns.barplot(x="Condition", y="mean", data=summary_stats, capsize=0.2)

# Add error bars manually
x_positions = range(len(summary_stats))
plt.errorbar(
    x=x_positions,
    y=summary_stats["mean"],
    yerr=summary_stats["ci_95"].values,  # Ensure it's a NumPy array
    fmt="none",
    capsize=5,
    color="black"
)

#plt.ylabel("Score (Mean ± 95% CI)", fontsize=fontsize_labels)
plt.ylabel("Score", fontsize=fontsize_labels)

plt.xlabel("", fontsize=fontsize_labels)
#plt.title("MUSHRA Mean Scores with 95% Confidence Intervals", fontsize=20)
plt.xticks(rotation=45, fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.grid()

# Add significance labels
for i, p in enumerate(ax.patches):
    ax.annotate(summary_stats["significance"].iloc[i],
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=15, color="red")

# Show the plots
plt.tight_layout()
plt.savefig("../results/histograms48/listening_test_scores.png")
plt.show()

# Save results to CSV
summary_stats.to_csv("mushra_summary.csv", index=False)
