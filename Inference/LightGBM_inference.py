import os
import pickle
import numpy as np
import pandas as pd
from glob import glob
import logging
import warnings

warnings.filterwarnings('ignore')

class DocumentInference:
    def __init__(self, models_dir):
        """
        Initialize the Inference pipeline by loading saved models.
        """
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Define the exact mapping used during training
        self.CLASS_MAPPING = {
            "Bill_of_Exchange": 0,
            "Bill_of_Lading": 1,
            "Certificate_of_Origin": 2,
            "Commercial_Invoice": 3,
            "Covering_Schedule": 4,
            "Others": 5,
            "Packing_List": 6
        }
        
        # Create a reverse mapping to convert numeric predictions back to string labels
        self.REVERSE_MAPPING = {v: k for k, v in self.CLASS_MAPPING.items()}
        
        # Load the models
        self.logger.info(f"Loading models from {models_dir}...")
        
        tfidf_path = os.path.join(models_dir, "tf_idf_model.pkl")
        lgb_path = os.path.join(models_dir, "lgb_model.pkl")
        
        if not os.path.exists(tfidf_path) or not os.path.exists(lgb_path):
            raise FileNotFoundError(f"Models not found in {models_dir}. Did you run the training script first?")
            
        with open(tfidf_path, "rb") as f:
            self.tf_idf = pickle.load(f)
            
        with open(lgb_path, "rb") as f:
            self.model = pickle.load(f)
            
        self.logger.info("Models loaded successfully.")

    def run_inference(self, image_folder, ocr_folder, output_csv):
        """
        Run predictions on a folder of images and save to CSV.
        """
        self.logger.info(f"Scanning for images in {image_folder}...")
        
        # Find all images recursively
        image_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"):
            pattern = os.path.join(image_folder, "**", ext)
            image_files.extend(glob(pattern, recursive=True))
            
        if not image_files:
            self.logger.error("No images found in the specified directory.")
            return
            
        self.logger.info(f"Found {len(image_files)} images to classify.")
        
        results = []
        
        for img_path in image_files:
            image_name = os.path.basename(img_path)
            
            # Calculate relative path to match the OCR folder structure
            rel_path = os.path.relpath(img_path, image_folder)
            rel_dir = os.path.dirname(rel_path)
            
            # Construct expected OCR filename
            base_name = os.path.splitext(image_name)[0]
            ocr_filename = os.path.join(ocr_folder, rel_dir, f"{base_name}_alltext.txt")
            
            extracted_text = ""
            
            # Read OCR text if it exists
            if os.path.exists(ocr_filename):
                with open(ocr_filename, 'r', encoding='utf-8') as f:
                    extracted_text = f.read().strip()
            
            # Handle missing or empty OCR gracefully
            if not extracted_text:
                self.logger.warning(f"No OCR text found for {image_name}. Marking as 'OCR Missing'.")
                results.append({
                    "Image_Name": image_name,
                    "Predicted_Class": "OCR Missing",
                    "Confidence_Score": 0.0
                })
                continue
            
            # Preprocess and Predict
            # TfidfVectorizer expects an iterable of strings
            X_input = self.tf_idf.transform([extracted_text])
            
            # LightGBM predict returns an array of probabilities for each class
            pred_probs = self.model.predict(X_input)[0] 
            
            # Get the highest probability index
            best_idx = np.argmax(pred_probs)
            confidence = pred_probs[best_idx]
            
            # Map index back to class name
            predicted_class = self.REVERSE_MAPPING[best_idx]
            
            results.append({
                "Image_Name": image_name,
                "Predicted_Class": predicted_class,
                "Confidence_Score": round(confidence, 4)
            })
            
        # Save to CSV
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        self.logger.info(f"Inference complete! Results saved to {output_csv}")
        
        # Print a quick summary
        self.logger.info("Prediction Summary:")
        self.logger.info(f"\n{df['Predicted_Class'].value_counts().to_string()}")


def main():
    # ------------------- Configuration -------------------
    # Paths based on your training environment
    base_dir = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification"
    
    models_dir = os.path.join(base_dir, "Data/LightGBM_train/models")
    
    # ⚠️ UPDATE THESE PATHS for inference
    # Where your new, unclassified images are stored
    unclassified_images_dir = os.path.join(base_dir, "Data/new_images_to_classify") 
    
    # Where the OCR text files for these new images are stored
    unclassified_ocr_dir = os.path.join(base_dir, "Data/new_ocr_data")
    
    # Where you want the final CSV to be saved
    output_csv = os.path.join(base_dir, "Inference/classification_results.csv")
    
    # ------------------- Execution -------------------
    inferencer = DocumentInference(models_dir)
    inferencer.run_inference(
        image_folder=unclassified_images_dir,
        ocr_folder=unclassified_ocr_dir,
        output_csv=output_csv
    )

if __name__ == "__main__":
    main()