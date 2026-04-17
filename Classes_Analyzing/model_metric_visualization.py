import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def parse_float(value):
    """
    Convert values like 'np.float64(0.999)' or numeric types to float.
    """
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        return float(value.replace("np.float64(", "").replace(")", ""))
    return 0.0


def extract_metrics(metrics_json):
    """
    Extract accuracy, precision, recall, and F1 metrics for
    train, validation, and test datasets.
    """
    token_cls = metrics_json["tokenClassification"]["b3"]

    def get_metric(metric_type):
        metric = token_cls.get(metric_type, [{}])[0]
        return {
            "accuracy": parse_float(metric.get("accuracy", 0)),
            "precision_micro": parse_float(metric.get("precisionMicroAverage", 0)),
            "precision_macro": parse_float(metric.get("precisionMacroAverage", 0)),
            "recall_micro": parse_float(metric.get("recallMicroAverage", 0)),
            "recall_macro": parse_float(metric.get("recallMacroAverage", 0)),
            "f1_micro": parse_float(metric.get("fMeasureMicroAverage", 0)),
            "f1_macro": parse_float(metric.get("fMeasureMacroAverage", 0)),
            "confusion_matrix": metric.get("confusionMatrix", [])
        }

    return {
        "train": get_metric("trainMetric"),
        "validation": get_metric("validationMetric"),
        "test": get_metric("testMetric")
    }


def build_confusion_matrix(confusion_data, class_labels):
    """
    Convert the confusionMatrix JSON into a NumPy array.
    """
    matrix = []
    row_labels = []

    for row in confusion_data:
        values = list(map(int, row["rowValues"].split(",")))
        matrix.append(values)
        row_labels.append(row["rowid"].replace("B-", ""))

    matrix = np.array(matrix)

    # Ensure matrix is square
    if matrix.shape[1] != len(row_labels):
        size = max(matrix.shape[0], matrix.shape[1])
        padded_matrix = np.zeros((size, size), dtype=int)
        padded_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
        matrix = padded_matrix
        row_labels += [f"Class_{i}" for i in range(len(row_labels), size)]

    return matrix, row_labels


def plot_confusion_matrix(ax, cm, labels):
    """
    Plot the confusion matrix on the given axis.
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")


def plot_metric_comparison(ax, metrics):
    """
    Plot comparison of evaluation metrics across Train, Validation, and Test.
    """
    metric_names = [
        "accuracy",
        "precision_micro",
        "precision_macro",
        "recall_micro",
        "recall_macro",
        "f1_micro",
        "f1_macro",
    ]

    datasets = ["train", "validation", "test"]
    x = np.arange(len(metric_names))
    width = 0.25

    for i, dataset in enumerate(datasets):
        values = [metrics[dataset][m] for m in metric_names]
        ax.bar(x + i * width, values, width, label=dataset.capitalize())

    ax.set_xticks(x + width)
    ax.set_xticklabels(
        [
            "Accuracy",
            "Prec μ",
            "Prec M",
            "Recall μ",
            "Recall M",
            "F1 μ",
            "F1 M",
        ],
        rotation=45,
    )
    ax.set_ylim(0, 1.05)
    ax.set_title("Metric Comparison (Train vs Validation vs Test)")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)


def plot_epoch_f1(ax, epoch_json):
    """
    Plot Epoch vs F1 Score.
    """
    series = epoch_json["series"][0]
    f1_scores = series["data"]
    epochs = epoch_json["xAxis"]["categories"]

    ax.plot(epochs, f1_scores, marker="o")
    ax.set_title("Epoch vs F1 Score")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.7)


def create_visualizations(metrics_path, epoch_path, output_path):
    """
    Main function to generate all three plots on a single page.
    """
    # Load JSON files
    with open(metrics_path, "r") as f:
        metrics_json = json.load(f)

    with open(epoch_path, "r") as f:
        epoch_json = json.load(f)

    # Extract metrics
    metrics = extract_metrics(metrics_json)

    # Get class labels
    class_labels = metrics_json["servingMetadataExtraction"]["classes"]

    # Use validation confusion matrix by default
    confusion_data = metrics["validation"]["confusion_matrix"]
    cm, labels = build_confusion_matrix(confusion_data, class_labels)

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    plot_confusion_matrix(axes[0], cm, labels)
    plot_metric_comparison(axes[1], metrics)
    plot_epoch_f1(axes[2], epoch_json)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate confusion matrix and metric comparison plots."
    )
    parser.add_argument(
        "--metrics_json",
        type=str,
        required=True,
        help="Path to the pipeline metrics JSON file.",
    )
    parser.add_argument(
        "--epoch_json",
        type=str,
        required=True,
        help="Path to the epoch vs F1 JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_evaluation_plots.png",
        help="Output image file path.",
    )

    args = parser.parse_args()

    create_visualizations(
        metrics_path=args.metrics_json,
        epoch_path=args.epoch_json,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()