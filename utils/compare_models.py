import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. Project paths
# =========================================================

CURRENT_FILE = os.path.abspath(__file__)
UTILS_FOLDER = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(UTILS_FOLDER)

OUTPUTS_FOLDER = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

LSA_OUTPUT_FOLDER = os.path.join(
    OUTPUTS_FOLDER,
    "lsa"
)

LSTM_OUTPUT_FOLDER = os.path.join(
    OUTPUTS_FOLDER,
    "lstm"
)

COMPARISON_OUTPUT_FOLDER = os.path.join(
    OUTPUTS_FOLDER,
    "comparison"
)

os.makedirs(
    COMPARISON_OUTPUT_FOLDER,
    exist_ok=True
)


# =========================================================
# 2. File paths
# =========================================================

LSA_ROUGE_PATH = os.path.join(
    LSA_OUTPUT_FOLDER,
    "lsa_rouge_scores.csv"
)

LSTM_ROUGE_PATH = os.path.join(
    LSTM_OUTPUT_FOLDER,
    "lstm_rouge_scores.csv"
)

LSTM_EVALUATION_PATH = os.path.join(
    LSTM_OUTPUT_FOLDER,
    "lstm_evaluation.csv"
)

COMPARISON_CSV_PATH = os.path.join(
    COMPARISON_OUTPUT_FOLDER,
    "model_comparison.csv"
)

ROUGE_GRAPH_PATH = os.path.join(
    COMPARISON_OUTPUT_FOLDER,
    "rouge_comparison.png"
)

LSTM_METRICS_GRAPH_PATH = os.path.join(
    COMPARISON_OUTPUT_FOLDER,
    "lstm_test_metrics.png"
)


# =========================================================
# 3. Check required files
# =========================================================

required_files = [
    LSA_ROUGE_PATH,
    LSTM_ROUGE_PATH,
    LSTM_EVALUATION_PATH
]

missing_files = [
    path
    for path in required_files
    if not os.path.exists(path)
]

if missing_files:

    print("=" * 70)
    print("ERROR: Required evaluation files are missing")
    print("=" * 70)

    for file_path in missing_files:
        print(file_path)

    print("\nRequired files:")
    print("outputs/lsa/lsa_rouge_scores.csv")
    print("outputs/lstm/lstm_rouge_scores.csv")
    print("outputs/lstm/lstm_evaluation.csv")

    print("\nRun LSA and LSTM evaluation scripts first.")

    raise SystemExit


# =========================================================
# 4. Load ROUGE results
# =========================================================

print("=" * 70)
print("Loading LSA and LSTM Evaluation Results")
print("=" * 70)

lsa_rouge_df = pd.read_csv(
    LSA_ROUGE_PATH
)

lstm_rouge_df = pd.read_csv(
    LSTM_ROUGE_PATH
)

lstm_evaluation_df = pd.read_csv(
    LSTM_EVALUATION_PATH
)


# =========================================================
# 5. Standardize column names
# =========================================================

lsa_rouge_df.columns = [
    column.strip()
    for column in lsa_rouge_df.columns
]

lstm_rouge_df.columns = [
    column.strip()
    for column in lstm_rouge_df.columns
]

lstm_evaluation_df.columns = [
    column.strip()
    for column in lstm_evaluation_df.columns
]


# =========================================================
# 6. Validate ROUGE columns
# =========================================================

required_rouge_columns = [
    "Metric",
    "Precision",
    "Recall",
    "F1 Score"
]

for column in required_rouge_columns:

    if column not in lsa_rouge_df.columns:
        raise ValueError(
            f"Missing column in LSA ROUGE file: {column}"
        )

    if column not in lstm_rouge_df.columns:
        raise ValueError(
            f"Missing column in LSTM ROUGE file: {column}"
        )


# =========================================================
# 7. Prepare LSA comparison data
# =========================================================

lsa_comparison = lsa_rouge_df[
    [
        "Metric",
        "Precision",
        "Recall",
        "F1 Score"
    ]
].copy()

lsa_comparison["Model"] = "LSA"


# =========================================================
# 8. Prepare LSTM comparison data
# =========================================================

lstm_comparison = lstm_rouge_df[
    [
        "Metric",
        "Precision",
        "Recall",
        "F1 Score"
    ]
].copy()

lstm_comparison["Model"] = "LSTM"


# =========================================================
# 9. Combine model results
# =========================================================

comparison_df = pd.concat(
    [
        lsa_comparison,
        lstm_comparison
    ],
    ignore_index=True
)

comparison_df = comparison_df[
    [
        "Model",
        "Metric",
        "Precision",
        "Recall",
        "F1 Score"
    ]
]


# =========================================================
# 10. Save comparison table
# =========================================================

comparison_df.to_csv(
    COMPARISON_CSV_PATH,
    index=False,
    encoding="utf-8"
)

print("\nModel comparison table:")
print(comparison_df.to_string(index=False))

print("\nComparison CSV saved to:")
print(COMPARISON_CSV_PATH)


# =========================================================
# 11. Create ROUGE F1 comparison table
# =========================================================

rouge_f1_pivot = comparison_df.pivot(
    index="Metric",
    columns="Model",
    values="F1 Score"
)

print("\n" + "=" * 70)
print("ROUGE F1 Comparison")
print("=" * 70)

print(
    rouge_f1_pivot.to_string()
)


# =========================================================
# 12. Plot ROUGE comparison graph
# =========================================================

ax = rouge_f1_pivot.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title(
    "LSA vs LSTM ROUGE F1 Score Comparison"
)

plt.xlabel(
    "ROUGE Metric"
)

plt.ylabel(
    "F1 Score"
)

plt.xticks(
    rotation=0
)

plt.legend(
    title="Model"
)

plt.tight_layout()

plt.savefig(
    ROUGE_GRAPH_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nROUGE comparison graph saved to:")
print(ROUGE_GRAPH_PATH)


# =========================================================
# 13. Read LSTM test metrics
# =========================================================

if (
    "Metric" not in lstm_evaluation_df.columns
    or "Value" not in lstm_evaluation_df.columns
):
    raise ValueError(
        "lstm_evaluation.csv must contain "
        "'Metric' and 'Value' columns."
    )

print("\n" + "=" * 70)
print("LSTM Test Metrics")
print("=" * 70)

print(
    lstm_evaluation_df.to_string(
        index=False
    )
)


# =========================================================
# 14. Plot LSTM accuracy and loss
# =========================================================

lstm_metrics = lstm_evaluation_df.copy()

lstm_metrics["Value"] = pd.to_numeric(
    lstm_metrics["Value"],
    errors="coerce"
)

lstm_metrics = lstm_metrics.dropna(
    subset=["Value"]
)

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    lstm_metrics["Metric"],
    lstm_metrics["Value"]
)

plt.title(
    "LSTM Test Accuracy and Loss"
)

plt.xlabel(
    "Metric"
)

plt.ylabel(
    "Value"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    LSTM_METRICS_GRAPH_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nLSTM metrics graph saved to:")
print(LSTM_METRICS_GRAPH_PATH)


# =========================================================
# 15. Identify best model
# =========================================================

average_f1_scores = comparison_df.groupby(
    "Model"
)["F1 Score"].mean()

best_model = average_f1_scores.idxmax()
best_score = average_f1_scores.max()


print("\n" + "=" * 70)
print("Overall Model Comparison")
print("=" * 70)

print("\nAverage ROUGE F1 Scores:")

for model_name, score in average_f1_scores.items():

    print(
        f"{model_name}: {score:.4f}"
    )


print(
    f"\nBest Performing Model: {best_model}"
)

print(
    f"Average ROUGE F1 Score: {best_score:.4f}"
)


# =========================================================
# 16. Save model conclusion
# =========================================================

CONCLUSION_PATH = os.path.join(
    COMPARISON_OUTPUT_FOLDER,
    "model_comparison_conclusion.txt"
)

with open(
    CONCLUSION_PATH,
    mode="w",
    encoding="utf-8"
) as file:

    file.write(
        "LSA and LSTM Model Comparison\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )

    file.write(
        "Average ROUGE F1 Scores:\n"
    )

    for model_name, score in average_f1_scores.items():

        file.write(
            f"{model_name}: {score:.4f}\n"
        )

    file.write(
        f"\nBest Performing Model: {best_model}\n"
    )

    file.write(
        f"Average ROUGE F1 Score: {best_score:.4f}\n"
    )


print("\nComparison conclusion saved to:")
print(CONCLUSION_PATH)


# =========================================================
# 17. Final output
# =========================================================

print("\n" + "=" * 70)
print("Model Comparison Completed Successfully")
print("=" * 70)

print("\nGenerated files:")

print(COMPARISON_CSV_PATH)
print(ROUGE_GRAPH_PATH)
print(LSTM_METRICS_GRAPH_PATH)
print(CONCLUSION_PATH)