# Extraction Benchmarking Report Generation

This README outlines the process of generating accuracy reports for Information Extraction predictions using the benchmarking codebase. The process evaluates the model's predicted JSON files against ground truth labels and automatically categorizes fields to apply context-aware fuzzy matching.

## Core Components

The report generation pipeline primarily depends on three scripts located in the `extrction_benchmarking_codebase/` directory:

1. **`report_generator_rupankar.py`**: The main entry point. It loads configurations, locates the prediction and ground-truth data, initializes the evaluation environment, and sequentially triggers the generation of overall label-wise, rules-dependent, and most-frequently occurred field reports.
2. **`label_category.py`**: Defines the `LabelCategory` class, which is responsible for holding the categories of different data fields (like numeric, date, multi-line, single-word, etc.). It parses field groupings from a configuration JSON file to map each extracted field to its respective type.
3. **`post_process_rupankar.py`**: Contains the core logic for calculating the extraction metrics. It compares ground truth to predictions, applying category-specific fuzzy matching ratios, numeric parsing, and bounding box checks. It also contains the functions that format, aggregate, and export the final metrics to CSV and text reports.

---

## Process Overview

1. **Initialization:** `report_generator_rupankar.py` reads `config.ini` and `prod.ini` to resolve which product/document needs to be processed.
2. **Field Categorization:** Using `BOE_post_process.json` (or the respective JSON mapped in setup), `label_category.py` parses all keys to know whether a given field is a single word, multi-line text, address, numeric, etc.
3. **Data Loading:** The script iterates through the Ground Truth (labels) files inside the defined `New_Master_Data_Merged` folder and looks for the matching prediction files in the `Results_Images` directory.
4. **Metric Calculation:** For every document and every extracted key, `post_process_rupankar.py` checks the ratio config (found in `config.ini`) mapped to that key's category. It compares the `actual` value against the `predicted` value. Special handling is implemented for numeric comparisons and string standardizations.
5. **Report Saving:** Several variants of CSV summaries and detailed label-match reports are created, along with STP (Straight Through Processing) summaries. They are categorized and output into timestamps under the `result_path`.

---

## 📂 Expected Input Directory Structure

To run the benchmarking code on a given dataset, your target folder path must be structured securely like this:

```text
📁 {Base_Document_Folder}                         <-- The 'folder_path' mapped in prod.ini
 ├── 📁 New_Master_Data_Merged                    <-- Contains your Ground Truth label files
 │    ├── sample_doc_1_labels.txt                 <-- Must contain JSON formatted annotations
 │    ├── sample_doc_2_labels.txt
 │    └── ...
 ├── 📁 Results_Images                            <-- Contains the prediction output files
 │    ├── sample_doc_1.json                       <-- Predictions corresponding to sample_doc_1
 │    ├── sample_doc_2.json
 │    └── ...
 └── 📁 result_path                               <-- Automatically generated: where reports are stored
      ├── 📁 label_wise
      ├── 📁 rules_required
      └── 📁 most_frequently
```

---

## 🛠️ How to Configure & Run on New Data

If you need to execute evaluating reports on a new dataset or different document type, follow these steps to adjust paths, configs, and JSON definitions.

### 1. Update the Post-Processing JSON (e.g., `BOE_post_process.json`)
This file is the main configuration map that instructs the evaluator on how to treat each key logically.
- When creating a new configuration JSON, sort all the expected extraction keys into the arrays matching their nature: `numeric_fields`, `address_fields`, `date_fields`, `single_line_fields`, `single_word_fields`, `multi_line_fields`, etc.
- **Why?** Since a multi-line generic text has a different fuzzy match requirement than a rigid numeric value, putting them under the right array ensures the codebase applies the right matching logic.
- Fields you do not wish to evaluate for accuracy (like `stamp` or `sign`) should be placed in `remove_fields`.

### 2. Update Configuration Files (`.ini`)
Inside `post_processing/config/` (or related config directories), update the following:

* **`config.ini` (Master config):** Ensure `Product`, `code` and `document_code` align with your new data run. 
* **`prod.ini`:** Make sure the combination of Product Code and Document Code correctly resolves to the absolute `{Base_Document_Folder}` containing your `New_Master_Data_Merged` and `Results_Images`.
* **Fuzzy Ratios & File Path Setup (`post_processing/config.ini`):**
   - In the `[PATH]` block, ensure pointers like `rules_required_keys` point to the respective actual rules JSONs.
   - You must add a pointer to your newly created Post-Processing JSON that maps the fields (e.g. `BOE_post_process.json`).
   - Under `[FuzzyRatio]`, adjust the numerical threshold required to accept a prediction as a "Match" for different category fields (like setting `address_fields = 85`).

### 3. Execution

Once the `.ini` config paths are wired correctly, your ground truth labels have the names ending in `labels.txt` holding valid JSON dicts, and prediction targets have `.json` or `1.txt` extensions, simply execute:

```bash
python extrction_benchmarking_codebase/report_generator_rupankar.py
```

The evaluator will calculate metrics, fuzz-match comparisons, and generate CSV reports detailing accuracies, count matching, and field-wise STP evaluations within your `result_path` dynamically named by timestamps.
