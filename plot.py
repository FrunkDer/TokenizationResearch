import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load classification data from a JSON file
def load_classification_data(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found. Skipping.")
        return None
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

# Function to save plots
def save_plot(fig, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)  # Close figure to free memory

# Function to plot and save class-wise F1-scores
def plot_f1_scores(data, lang, strat, save_dir):
    classification_report = data["classification_report"]
    
    labels = [label for label in classification_report.keys() if label not in ["accuracy", "macro avg", "weighted avg"]]
    f1_scores = [classification_report[label]["f1-score"] for label in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels, f1_scores, color='blue')
    ax.set_xlabel("F1-Score")
    ax.set_ylabel("Class Labels")
    ax.set_title(f"{lang} {strat} Class-wise F1-Scores")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    save_plot(fig, os.path.join(save_dir, "f1_scores.png"))

# Function to plot and save precision vs recall scatter plot
def plot_precision_vs_recall(data, lang, strat, save_dir):
    classification_report = data["classification_report"]
    
    labels = [label for label in classification_report.keys() if label not in ["accuracy", "macro avg", "weighted avg"]]
    precision = [classification_report[label]["precision"] for label in labels]
    recall = [classification_report[label]["recall"] for label in labels]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(precision, recall, color='red', s=100, alpha=0.7)
    
    for i, label in enumerate(labels):
        ax.text(precision[i], recall[i], label, fontsize=9, ha='right')

    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title(f"{lang} {strat} Precision vs Recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)

    save_plot(fig, os.path.join(save_dir, "precision_recall.png"))

# Function to plot and save confusion matrix heatmap
def plot_confusion_matrix(data, lang, strat, save_dir):
    confusion_matrix = np.array(data["confusion_matrix"])
    labels = data["labels"]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title(f"{lang} {strat} Confusion Matrix Heatmap")

    save_plot(fig, os.path.join(save_dir, "confusion_matrix.png"))

# Define base output directory
output_directory = r"C:\Users\jinfa\OneDrive\Desktop\Research Dr. Mani\Evaluation\Plots"  # Change this to your desired output folder

# Define languages and tokenization strategies
languages = ["Turkish", "Finnish"]
strategies = ["Bigrams", "BPE5k", "BPE10k", "BPE25k", "BPE50k", "Char", "Trigrams", "Word"]

# Loop through language and strategy combinations
for lang in languages:
    for strategy in strategies:
        file_path = os.path.join("Evaluation", f"{lang}_{strategy}_POS_results.json")  # Path to input JSON file
        
        data = load_classification_data(file_path)
        if data:  # Only process if file exists and is loaded
            save_dir = os.path.join(output_directory, lang, strategy)  # Define where to save plots

            plot_f1_scores(data, lang, strategy, save_dir)
            plot_precision_vs_recall(data, lang, strategy, save_dir)
            plot_confusion_matrix(data, lang, strategy, save_dir)

            print(f"Saved plots for {lang} - {strategy} in {save_dir}")
