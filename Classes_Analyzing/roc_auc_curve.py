import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# 1. Load the ODS file
file_path = '/home/lpt6964/Downloads/temp/groundtruth.ods'
df = pd.read_excel(file_path, engine='odf')

# 2. Extract Data
all_y_true = []
all_y_scores = []
all_y_pred = []

for i in range(1, len(df.columns), 3):
    gt_col = df.columns[i]
    pred_col = df.columns[i+1]
    conf_col = df.columns[i+2]
    
    for idx, row in df.iterrows():
        gt = row[gt_col]
        pred = row[pred_col]
        conf = row[conf_col]
        
        if pd.isna(gt) or pd.isna(pred) or pd.isna(conf):
            continue
            
        try:
            conf_float = float(conf)
        except ValueError:
            continue
            
        if abs(conf_float - 0.9) < 1e-5:
            continue
            
        gt_str = str(gt).strip()
        if gt_str.lower() in ('other', 'others'):
            gt_str = 'Others'
            
        pred_str = str(pred).strip()
        if pred_str.lower() in ('other', 'others'):
            pred_str = 'Others'
            
        all_y_true.append(gt_str)
        all_y_pred.append(pred_str)
        all_y_scores.append(conf_float)

# 3. Prepare data for Multi-class ROC AUC
classes = sorted(list(set(all_y_true).union(set(all_y_pred))))

y_true_bin = label_binarize(all_y_true, classes=classes)
n_classes = y_true_bin.shape[1]

y_score_matrix = np.zeros((len(all_y_true), n_classes))

for i, (pred_label, conf) in enumerate(zip(all_y_pred, all_y_scores)):
    if pred_label in classes:
        pred_idx = classes.index(pred_label)
        remaining_conf = max(0, 1.0 - conf)
        if n_classes > 1:
            per_class_remaining = remaining_conf / (n_classes - 1)
            y_score_matrix[i, :] = per_class_remaining
            y_score_matrix[i, pred_idx] = conf
        else:
            y_score_matrix[i, pred_idx] = conf

# 4. Compute ROC curve, AUC, and optimal thresholds for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
thresholds = dict()

print("\n--- OPTIMAL THRESHOLDS (Youden's J Index) ---")
best_points = dict()

for i in range(n_classes):
    fpr[i], tpr[i], thresholds[i] = roc_curve(y_true_bin[:, i], y_score_matrix[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    J = tpr[i] - fpr[i]
    best_idx = np.argmax(J)
    best_thresh = thresholds[i][best_idx]
    
    best_points[i] = {
        'threshold': best_thresh,
        'fpr': fpr[i][best_idx],
        'tpr': tpr[i][best_idx]
    }
    
    print(f"Class '{classes[i]}': Threshold = {best_thresh:.4f} (TPR: {tpr[i][best_idx]:.4f}, FPR: {fpr[i][best_idx]:.4f})")
print("---------------------------------------------\n")

# Compute micro-average ROC curve, ROC area, and Generalized Threshold
fpr["micro"], tpr["micro"], thresholds["micro"] = roc_curve(y_true_bin.ravel(), y_score_matrix.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

J_micro = tpr["micro"] - fpr["micro"]
best_idx_micro = np.argmax(J_micro)
best_thresh_micro = thresholds["micro"][best_idx_micro]

print(f"--- GENERALIZED OPTIMAL THRESHOLD (Micro-Average) ---")
print(f"Global Threshold = {best_thresh_micro:.4f} (Global TPR: {tpr['micro'][best_idx_micro]:.4f}, Global FPR: {fpr['micro'][best_idx_micro]:.4f})")
print("-----------------------------------------------------\n")


# 5. Plot ROC Curve
plt.figure(figsize=(12, 9))
plt.plot(fpr["micro"], tpr["micro"],
         label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
         color='deeppink', linestyle=':', linewidth=4)

# Plot the generalized optimal point on the micro-average curve
plt.scatter(fpr["micro"][best_idx_micro], tpr["micro"][best_idx_micro], s=250, color='deeppink', edgecolors='black', zorder=6, marker='*')
plt.annotate(f"Global Thresh: {best_thresh_micro:.2f}", 
             (fpr["micro"][best_idx_micro], tpr["micro"][best_idx_micro]),
             textcoords="offset points", 
             xytext=(-15, 15),
             ha='right', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="deeppink", lw=1.5, alpha=0.9))

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:0.2f})')
    
    bp = best_points[i]
    plt.scatter(bp['fpr'], bp['tpr'], s=100, color=color, edgecolors='black', zorder=5)
    
    plt.annotate(f"{classes[i]}: {bp['threshold']:.2f}", 
                 (bp['fpr'], bp['tpr']),
                 textcoords="offset points", 
                 xytext=(10, -10),
                 ha='left', fontsize=10, 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5, alpha=0.8))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (False Alarms)', fontsize=12)
plt.ylabel('True Positive Rate (Correct Calls)', fontsize=12)
plt.title('Multi-class ROC AUC Curve with Optimal Thresholds', fontsize=14)
plt.legend(loc="lower right", fontsize='small')
plt.grid(alpha=0.3)
plt.tight_layout()

# Save the plot
output_plot_path = '/home/lpt6964/Downloads/temp/roc_auc_curve_marked.png'
plt.savefig(output_plot_path, dpi=300)
print(f"ROC AUC curve saved to {output_plot_path}")
plt.show()
