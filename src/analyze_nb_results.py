import pandas as pd
import matplotlib.pyplot as plt

# Load metrics CSV
df = pd.read_csv("results/naive_bayes_metrics.csv")

# Identify metric columns
acc_cols = [c for c in df.columns if "_acc" in c]
prec_cols = [c for c in df.columns if "_prec" in c]
rec_cols = [c for c in df.columns if "_rec" in c]
f1_cols = [c for c in df.columns if "_f1" in c]

# Compute average for each metric per hyperparameter combination
df["avg_acc"] = df[acc_cols].mean(axis=1)
df["avg_precision"] = df[prec_cols].mean(axis=1)
df["avg_recall"] = df[rec_cols].mean(axis=1)
df["avg_f1"] = df[f1_cols].mean(axis=1)

# Sort by average F1 (or any other metric)
df_sorted = df.sort_values("avg_f1", ascending=False)

# Print top 5 combinations
print("\nTop 5 hyperparameter combinations (averaged metrics):")
print(df_sorted[["ngram_range","alpha","avg_acc","avg_precision","avg_recall","avg_f1"]].head())

# Optional: Plot all metrics for the best combination
best_row = df_sorted.iloc[0]
emotions = [c.replace("_f1","") for c in f1_cols]
metrics = {
    "Accuracy": best_row[[c for c in acc_cols]].values,
    "Precision": best_row[[c for c in prec_cols]].values,
    "Recall": best_row[[c for c in rec_cols]].values,
    "F1-score": best_row[[c for c in f1_cols]].values
}

plt.figure(figsize=(15,5))
for metric_name, values in metrics.items():
    plt.plot(emotions, values, marker='o', label=metric_name)

plt.xticks(rotation=90)
plt.ylabel("Score")
plt.title(f"All metrics per emotion for best combo: ngram={best_row['ngram_range']}, alpha={best_row['alpha']}")
plt.legend()
plt.tight_layout()
plt.show()
