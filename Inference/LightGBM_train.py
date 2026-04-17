import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix
)
import lightgbm as lgb
import pickle
import logging
import json
import warnings
import time
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings('ignore')

class FinancialDocumentClassifier:
    def __init__(self, data_dir, ocr_dir, results_dir):
        """
        Initialize the Financial Document Classifier
        
        Args:
            data_dir (str): Directory containing document images
            ocr_dir (str): Directory containing pre-extracted OCR text results
            results_dir (str): Directory to store classification results
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.data_dir = data_dir
        self.ocr_dir = ocr_dir
        self.results_dir = results_dir
        
        # Mapping of folder names to labels
        self.CLASS_MAPPING = {
    "Bill_of_Exchange": 0,
    "Bill_of_Lading": 1,
    "Certificate_of_Origin": 2,
    "Commercial_Invoice": 3,
    "Covering_Schedule": 4,
    "Others": 5,
    "Packing_List": 6
}
        
        # Ensure directories exist
        for directory in [self.ocr_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
            for class_name in self.CLASS_MAPPING.keys():
                os.makedirs(os.path.join(directory, class_name), exist_ok=True)
    
    def process_images(self):
        """
        Process images for all classes by reading their corresponding pre-extracted OCR text
        
        Returns:
            list: List of processed document records
        """
        all_records = []
        
        # Process each class
        for class_name, label in self.CLASS_MAPPING.items():
            # Construct full path for class
            class_path = os.path.join(self.data_dir, class_name)
            
            # Get image files
            image_files = self.get_image_files(class_path)
            self.logger.info(f"Processing {class_name}: {len(image_files)} images found")
            
            # Process each image
            for img_path in image_files:
                # Calculate relative path from data_dir to match the OCR folder structure
                rel_path = os.path.relpath(img_path, self.data_dir)
                rel_dir = os.path.dirname(rel_path)
                
                # Extract base filename and format as the OCR output
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                ocr_filename = os.path.join(self.ocr_dir, rel_dir, f"{base_name}_alltext.txt")
                
                # Check if the OCR file exists and read it
                if os.path.exists(ocr_filename):
                    with open(ocr_filename, 'r', encoding='utf-8') as f:
                        extracted_text = f.read()
                        
                    # Append record ONLY if the file exists
                    all_records.append({
                        "image_data": os.path.basename(img_path),
                        "classes": label,
                        "OCR": extracted_text.split(),
                        "class_name": class_name
                    })
                else:
                    # Skip to the next image if OCR is missing
                    self.logger.warning(f"Missing OCR text file for: {rel_path}. Skipping image.")
                    continue
        
        return all_records
    
    def get_image_files(self, class_path):
        """
        Get image files from a given class path recursively
        
        Args:
            class_path (str): Path to the class directory
        
        Returns:
            list: List of image file paths
        """
        if not os.path.exists(class_path):
            self.logger.error(f"Image folder '{class_path}' does not exist!")
            return []
        
        image_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"):
            # Update to use ** and recursive=True for nested subfolders
            pattern = os.path.join(class_path, "**", ext)
            image_files.extend(glob(pattern, recursive=True))
        
        return image_files
    
    def train_lightgbm_classifier(self, df):
        """
        Train LightGBM classifier using TF-IDF vectorization and measure training/inference time
        """
        self.logger.info("Starting LightGBM classifier training process")
        
        # Preprocess OCR text
        def preprocess_text(ocr_list):
            return " ".join(ocr_list) if isinstance(ocr_list, list) else str(ocr_list)
        
        # Vectorize text
        self.logger.info("Vectorizing text using TF-IDF")
        tf_idf = TfidfVectorizer(
            stop_words='english', 
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        # Prepare text for vectorization
        processed_text = df["OCR"].apply(preprocess_text)
        start_time = time.time()
        X = tf_idf.fit_transform(processed_text)
        y = df["classes"]
        image_names = df["image_data"]
        self.logger.info(f"Text vectorization completed in {time.time() - start_time:.2f} seconds")
        self.logger.info(f"Feature matrix shape: {X.shape}")
        
        # Validate input data
        self.logger.info("Validating input data")
        X_array = X.toarray()
        if np.any(np.isnan(X_array)) or np.any(np.isinf(X_array)):
            self.logger.error("Input data contains NaN or infinite values")
            raise ValueError("Input data contains NaN or infinite values")
        if not all(y.isin(self.CLASS_MAPPING.values())):
            self.logger.error("Invalid class labels found in data")
            raise ValueError("Invalid class labels found")
        self.logger.info(f"Class distribution:\n{y.value_counts().to_string()}")
        
        # Split data
        self.logger.info("Splitting data into training and test sets")
        start_time = time.time()
        X_train, X_test, y_train, y_test, image_names_train, image_names_test = train_test_split(
            X, y, image_names,
            test_size=0.2, 
            stratify=y, 
            random_state=2023
        )
        self.logger.info(f"Data splitting completed in {time.time() - start_time:.2f} seconds")
        self.logger.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
        
        # Save train-test split to CSV
        self.logger.info("Saving train-test split to CSV")
        train_split_df = pd.DataFrame({
            'image_name': image_names_train,
            'label': y_train,
            'set': 'train'
        })
        test_split_df = pd.DataFrame({
            'image_name': image_names_test,
            'label': y_test,
            'set': 'test'
        })
        split_df = pd.concat([train_split_df, test_split_df], ignore_index=True)
        split_csv_path = os.path.join(self.results_dir, "train_test_split.csv")
        split_df.to_csv(split_csv_path, index=False)
        self.logger.info(f"Train-test split saved to {split_csv_path}")
        
        # LightGBM parameters
        params = {
            'objective': 'multiclass',
            'num_class': len(self.CLASS_MAPPING),
            'metric': 'multi_logloss',
            'verbose': -1,  # Set to -1 to reduce terminal spam
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        
        # # Create LightGBM datasets
        # self.logger.info("Creating LightGBM datasets")
        # train_data = lgb.Dataset(X_train, label=y_train)
        # valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Create LightGBM datasets
        self.logger.info("Creating LightGBM datasets with balanced weights")
        
        # Calculate weights to penalize mistakes on minority classes more heavily
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        
        # Pass the weights into the training dataset
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Measure training time
        self.logger.info("Starting LightGBM training")
        start_time = time.time()
        # Train LightGBM model
        num_round = 100
        try:
            model = lgb.train(
                params,
                train_data,
                num_round,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True)]
            )
            self.logger.info(f"LightGBM training completed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            self.logger.error(f"LightGBM training error: {e}")
            raise
        training_time = time.time() - start_time
        
        # Measure inference time
        self.logger.info("Generating predictions for training set")
        start_time = time.time()
        y_pred_train = model.predict(X_train).argmax(axis=1)
        train_inference_time = time.time() - start_time
        
        self.logger.info("Generating predictions for test set")
        start_time = time.time()
        y_pred_test = model.predict(X_test).argmax(axis=1)
        test_inference_time = time.time() - start_time
        
        # Detailed metrics with target names
        self.logger.info("Computing evaluation metrics")
        target_names = [
            list(self.CLASS_MAPPING.keys())[list(self.CLASS_MAPPING.values()).index(label)] 
            for label in np.unique(y)
        ]
        
        # Detailed metrics
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_pred_train),
            'classification_report': classification_report(
                y_train, y_pred_train, 
                target_names=target_names, 
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_train, y_pred_train)
        }
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'classification_report': classification_report(
                y_test, y_pred_test, 
                target_names=target_names, 
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test)
        }
        
        # Timing metrics
        timing_metrics = {
            'training_time_seconds': training_time,
            'train_inference_time_seconds': train_inference_time,
            'test_inference_time_seconds': test_inference_time
        }
        
        # Prediction results for CSV
        prediction_results = {
            'image_names_test': image_names_test,
            'y_test': y_test,
            'y_pred_test': y_pred_test
        }
        
        self.logger.info("LightGBM classifier training process completed")
        return tf_idf, model, (train_metrics, test_metrics), (X_train, X_test, y_train, y_test), timing_metrics, prediction_results

    def save_models(self, tf_idf, lgb_model, models_dir, train_metrics, test_metrics, timing_metrics, prediction_results):
        """
        Save TF-IDF and LightGBM models along with detailed evaluation results, timing metrics,
        and prediction results CSV
        """
        self.logger.info("Saving models and results")
        os.makedirs(models_dir, exist_ok=True)
        
        # Save TF-IDF vectorizer
        tf_idf_path = os.path.join(models_dir, "tf_idf_model.pkl")
        with open(tf_idf_path, "wb") as f:
            pickle.dump(tf_idf, f)
        
        # Save LightGBM model as text
        lgb_model_path = os.path.join(models_dir, "lgb_model.txt")
        lgb_model.save_model(lgb_model_path)
        
        # Save LightGBM model as pickle
        lgb_pickle_path = os.path.join(models_dir, "lgb_model.pkl")
        with open(lgb_pickle_path, "wb") as f:
            pickle.dump(lgb_model, f)
        
        # Helper function to convert numpy arrays to lists
        def convert_to_serializable(obj):
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Prepare metrics for JSON serialization
        serializable_metrics = {
            "train_metrics": {
                key: convert_to_serializable(value) 
                for key, value in train_metrics.items()
            },
            "test_metrics": {
                key: convert_to_serializable(value) 
                for key, value in test_metrics.items()
            },
            "timing_metrics": {
                key: convert_to_serializable(value)
                for key, value in timing_metrics.items()
            }
        }
        
        # Save metrics to JSON
        metrics_path = os.path.join(models_dir, "model_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(serializable_metrics, f, indent=4)
        
        # Save prediction results to CSV
        results_df = pd.DataFrame({
            'image_name': prediction_results['image_names_test'],
            'label': prediction_results['y_test'],
            'predicted_label': prediction_results['y_pred_test']
        })
        results_csv_path = os.path.join(models_dir, "prediction_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        
        self.logger.info("Models and metrics saved successfully")
        
        return models_dir
    
    def plot_classification_results(self, train_metrics, test_metrics, results_dir):
        """
        Generate and save detailed visualizations for classification results
        """
        self.logger.info("Generating classification result visualizations")
        os.makedirs(results_dir, exist_ok=True)
        
        # 1. Detailed Metrics Breakdown
        plt.figure(figsize=(15, 10))
        
        # Prepare data
        classes = list(self.CLASS_MAPPING.keys())
        metrics = ['precision', 'recall', 'f1-score']
        
        # Create subplot grid
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 15))
        fig.suptitle('Detailed Classification Metrics', fontsize=16)
        
        for i, metric in enumerate(metrics):
            # Extract metric values for each class
            train_values = [train_metrics['classification_report'].get(cls, {}).get(metric, 0) for cls in classes]
            test_values = [test_metrics['classification_report'].get(cls, {}).get(metric, 0) for cls in classes]
            
            # Plot
            x = np.arange(len(classes))
            width = 0.35
            
            axes[i].bar(x - width/2, train_values, width, label='Training Set', color='blue', alpha=0.7)
            axes[i].bar(x + width/2, test_values, width, label='Test Set', color='orange', alpha=0.7)
            
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_title(f'{metric.capitalize()} Scores')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(classes, rotation=45)
            axes[i].legend()
            
            # Add value labels
            for j, (train_val, test_val) in enumerate(zip(train_values, test_values)):
                axes[i].text(j - width/2, train_val, f'{train_val:.2f}', ha='center', va='bottom')
                axes[i].text(j + width/2, test_val, f'{test_val:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'detailed_metrics_breakdown.png'))
        plt.close()
        
        # 2. Confusion Matrix Visualization
        plt.figure(figsize=(15, 6))
        
        # Training Confusion Matrix
        plt.subplot(121)
        sns.heatmap(train_metrics['confusion_matrix'], 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues', 
                    xticklabels=classes, 
                    yticklabels=classes)
        plt.title('Confusion Matrix - Training Set')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Testing Confusion Matrix
        plt.subplot(122)
        sns.heatmap(test_metrics['confusion_matrix'], 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues', 
                    xticklabels=classes, 
                    yticklabels=classes)
        plt.title('Confusion Matrix - Test Set')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'confusion_matrices.png'))
        plt.close()
        self.logger.info(f"Classification result visualizations saved in {results_dir}")


def main():
    # Base directory configuration
    #base_dir = "/home/ng6281/Rupankar_Dev/Extraction_codebase/split_corrected_implementation"
    
    # Specific directory paths
    data_dir = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Raw_ouput_data/FINAL_merged_classified_images"
    
    # ⚠️ NOTE: Ensure this points to the abbyy_formatted_ocr directory
    ocr_dir = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/abby_clean_ocr"
    
    models_dir = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/LightGBM_train/models"
    results_dir = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/LightGBM_train/results"
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Output CSV path
    output_csv = os.path.join(results_dir, "financial_docs.csv")
    
    # Initialize classifier
    classifier = FinancialDocumentClassifier(data_dir, ocr_dir, results_dir)
    
    # Process images and create DataFrame
    classifier.logger.info("Starting image processing")
    records = classifier.process_images()
    
    # Check if we successfully read any records before proceeding
    if not records:
        classifier.logger.error("No OCR data could be found. Exiting.")
        return
        
    df = pd.DataFrame(records)
    
    # Save DataFrame to CSV
    df.to_csv(output_csv, index=False)
    classifier.logger.info(f"DataFrame saved to {output_csv}")
    classifier.logger.info("Class Distribution:")
    classifier.logger.info(f"\n{df['classes'].value_counts().to_string()}")
    
    # Train LightGBM Classifier
    classifier.logger.info("Starting LightGBM classifier training")
    tf_idf, lgb_model, (train_metrics, test_metrics), (X_train, X_test, y_train, y_test), timing_metrics, prediction_results = classifier.train_lightgbm_classifier(df)
    
    # Save models
    classifier.logger.info("Saving models and results")
    classifier.save_models(
        tf_idf, 
        lgb_model, 
        models_dir, 
        train_metrics, 
        test_metrics,
        timing_metrics,
        prediction_results
    )
    
    # Generate and save visualizations
    classifier.logger.info("Generating visualizations")
    classifier.plot_classification_results(
        train_metrics, 
        test_metrics, 
        results_dir
    )
    
    # Print detailed training and testing metrics
    classifier.logger.info("Training Set Metrics:")
    classifier.logger.info(f"\n{classification_report(y_train, lgb_model.predict(X_train).argmax(axis=1), target_names=list(classifier.CLASS_MAPPING.keys()))}")
    
    classifier.logger.info("Testing Set Metrics:")
    classifier.logger.info(f"\n{classification_report(y_test, lgb_model.predict(X_test).argmax(axis=1), target_names=list(classifier.CLASS_MAPPING.keys()))}")

if __name__ == "__main__":
    main()