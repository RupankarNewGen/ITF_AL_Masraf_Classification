# Inference 

## Main File

To run the main processing script, execute the following command:

```bash
python test_geo_images.py
```

### Required Paths

In the `test_geo_images.py` file, ensure the following paths are set:

- Images and OCR Paths:
  - `images_path`: This variable defines the path to the directory of images for Inference.
  - `OCR_path`: This variable specifies the path to the directory containing OCR (Optical Character Recognition)

### Results convertion

To convert results to expected format
```bash
python result_conversion.py
```
### Input and Output Folders

- **Input Folder**: This directory contains the raw results that need to be processed.
  ```
  /home/ntlpt19/Desktop/TF_release/geolm_api/grasim_test_samples/itter5/results
  ```

- **Output Folder**: This directory is where the converted results will be saved.
  ```
  /home/ntlpt19/Desktop/TF_release/geolm_api/grasim_test_samples/itter5/final_results
  ```
- **Note**:  Before running the conversion process, ensure that any files prefixed with _linking are deleted from the input folder. These files may interfere with the conversion process and should not be included.

## Configuration File

Ensure the `config.ini` file contains the following settings:

### Key Settings:

- `GEO_CLASSES_PATH`: Path to the class names used for the geo layout model.
- `GEO_DUMP`: Directory where results will be saved.
- `MODEL_PATH`: Path to the trained model file.

## Accuracy Report Generation

To generate an accuracy report, use the `Accuracy_report_gen_geo.py` file. Set the paths as follows:

```python
idp_inv_images_folder = "/home/ntlpt19/Desktop/TF_release/geolm_api/grasim_test_samples/Images"
idp_inv_labels_folder = "/home/ntlpt19/Desktop/TF_release/geolm_api/grasim_test_samples/Labels"
idp_inv_ocr_folder = "/home/ntlpt19/Desktop/TF_release/geolm_api/grasim_test_samples/OCR"
classes_path = "/home/ntlpt19/Desktop/TF_release/geolm_api/grasim_test_samples/class_names.txt"
annot_classses_file = "/home/ntlpt19/Desktop/TF_release/geolm_api/grasim_test_samples/label.txt"
idp_inv_json_results = "/home/ntlpt19/Desktop/TF_release/geolm_api/grasim_test_samples/itter5/final_results"
idp_inv_image_results = "/home/ntlpt19/Desktop/TF_release/geolm_api/grasim_test_samples/itter5/reports"
csv_file_path = './Inv_geo_OUTPUT_nov9'
plot_gt_flag = False
idp_model_type = "GEOlayoutLMVForTokenClassification"
```

### Explanation of Parameters:

- `idp_inv_images_folder`: Directory containing input images.
- `idp_inv_labels_folder`: Directory with ground truth labels.
- `idp_inv_ocr_folder`: Directory with OCR.
- `classes_path`: Path to the txt file containing class names.
- `annot_classses_file`: File containing annotation classes.
- `idp_inv_json_results`: Directory where JSON results will be stored.
- `idp_inv_image_results`: Directory for image results outputs  with bbox drawn, will be stored..
- `csv_file_path`: Path where the CSV output will be saved.
- `plot_gt_flag`: Flag for plotting ground truth.
- `idp_model_type`: Type of model used for classification.

## Dependencies

Make sure you have the necessary libraries installed. Typically, include instructions for libraries like TensorFlow, PyTorch, OpenCV, etc., depending on your project's requirements.

```bash
pip install -r requirements.txt
```
