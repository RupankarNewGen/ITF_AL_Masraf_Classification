import pandas as pd

def calculate_accuracy_v2(csv_path):
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Normalization function: removes spaces, underscores, and case
    def normalize(label):
        if pd.isna(label): return ""
        return str(label).lower().replace(" ", "").replace("_", "").strip()

    # Apply normalization to both columns
    df['norm_class'] = df['class'].apply(normalize)
    df['norm_truth'] = df['Ground Truth'].apply(normalize)
    
    # Comparison
    df['is_correct'] = df['norm_class'] == df['norm_truth']

    # 1. Overall Accuracy
    overall_acc = df['is_correct'].mean() * 100

    # 2. Class-level Accuracy 
    # Grouping by 'class' instead of 'Ground Truth' to consolidate variations
    class_stats = df.groupby('norm_class').agg(
        original_label=('class', 'first'), # Keep the readable name from the class column
        total_samples=('is_correct', 'count'),
        correct_predictions=('is_correct', 'sum')
    )
    class_stats['accuracy_pct'] = (class_stats['correct_predictions'] / class_stats['total_samples']) * 100

    # 3. Print the Cleaned Report
    print("\n" + "="*55)
    print(f"{'CONSOLIDATED CLASSIFICATION REPORT':^55}")
    print("="*55)
    print(f"OVERALL ACCURACY: {overall_acc:.2f}%")
    print("-" * 55)
    
    print(f"{'MASTER CLASS':<25} | {'COUNT':<6} | {'ACCURACY':<10}")
    print("-" * 55)
    
    # Sorting by count to see where most of your data sits
    for _, row in class_stats.sort_values(by='total_samples', ascending=False).iterrows():
        label = row['original_label']
        print(f"{str(label):<25} | {int(row['total_samples']):<6} | {row['accuracy_pct']:>8.2f}%")
    
    print("="*55)

if __name__ == "__main__":
    CSV_FILE = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Prediction_CSV/second_set_1000_images_with_ground_truth.csv"
    calculate_accuracy_v2(CSV_FILE)