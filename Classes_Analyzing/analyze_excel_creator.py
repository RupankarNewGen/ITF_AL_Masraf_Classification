import os
import json
import pandas as pd
from collections import defaultdict
import re


MODE = "class" # "doc/class"

# ================= LOAD CLASSES =================
def load_class_labels(root_dir):
    class_map = {}

    for cls in os.listdir(root_dir):
        cls_path = os.path.join(root_dir, cls)
        class_file = os.path.join(cls_path, "classes.txt")

        class_name = cls.split("_")[0]

        if os.path.exists(class_file):
            with open(class_file, "r") as f:
                class_map[class_name] = [line.strip() for line in f if line.strip()]

    return class_map

def load_gt(ods_file):
    df = pd.read_excel(ods_file, engine="odf")

    gt_map = {}
    
    for col in df.columns[1:]:
        raw_name = col.strip()
        norm_name = normalize_doc_name(raw_name)
        print(norm_name)
        if norm_name not in gt_map:
            gt_map[norm_name] = {}
        else:
            print(f"⚠️ Duplicate GT column mapped to same key: {norm_name} (merging)")

        for _, row in df.iterrows():

            page_str = str(row["Page Number"]).strip()

            match = re.search(r'(\d+)', page_str)
            if not match:
                continue

            page = int(match.group(1))
            gt_class = row[col]

            if pd.notna(gt_class):
                gt_map[norm_name][page] = str(gt_class).strip()

    print(f"\n✅ Total GT Docs Loaded: {len(gt_map)}")
    return gt_map


# ================= MERGE BBOX =================
def merge_bboxes(bboxes):
    merged = []
    for b in bboxes:
        merged.extend(b)
    return merged


def evaluate_document(df):

    # take unique page-level rows
    page_df = df[["Page", "GT Class", "Predicted Class"]].drop_duplicates()

    total = len(page_df)

    correct = (page_df["GT Class"] == page_df["Predicted Class"]).sum()
    incorrect = total - correct

    accuracy = correct / total if total > 0 else 0

    return correct, incorrect, total, accuracy


# ================= PROCESS =================
def process_document(doc_name, result_json, gt_data, class_map):

    rows = []

    for page_name, page_data in result_json.items():

        # 🔍 extract page index from ANY format like:
        # input_tif_file__0.png
        # input_pdf_file__0
        match = re.search(r"__(\d+)", page_name)

        if not match:
            print(f"⚠️ Skipping invalid key: {page_name}")
            continue

        pred_page_index = int(match.group(1))

        # ✅ convert to GT format
        gt_page_num = pred_page_index + 1
        page_label = f"Page_{gt_page_num}"
        gt_class = gt_data.get(doc_name, {}).get(gt_page_num, "")
        gt_text_map = load_gt_text(CLASS_ROOT, doc_name, gt_page_num)

        pred_class = page_data.get("predicted_class", "")
        confidence = page_data.get("confidence", "")

        extraction = page_data.get("extraction_result", {})

        # 🔥 labels from GT class
        if str(gt_class).lower() == "others":
            labels = []
            # print(f"⚠️ GT class is 'Others' for {doc_name} {page_label}. Using all extracted labels.")
        else:
            labels = class_map.get(gt_class, list(extraction.keys()))

        label_data = defaultdict(lambda: {"text": [], "bbox": [], "model_confidence": []})
        for label, details in extraction.items():

            if not isinstance(details, dict):
                continue

            value = details.get("value", "")
            bbox = details.get("coordinate", [])
            model_conf = details.get("model_confidence", "")

            if value:
                label_data[label]["text"].append(str(value))

            if bbox:
                label_data[label]["bbox"].append(bbox)
            
            if model_conf:
                label_data[label]["model_confidence"].append(model_conf)

        for label in labels:
            model_conf_list = label_data.get(label, {}).get("model_confidence", [])

            # handle different cases
            if isinstance(model_conf_list, list) and model_conf_list:
                model_conf = max(model_conf_list)   # or use avg if you want
            else:
                model_conf = ""


            rows.append({
                "Page": page_label,
                "Confidence": confidence,
                "GT Class": gt_class,
                "Predicted Class": pred_class,
                "Label": label,
                "GT Text": gt_text_map.get(label, ""),   # ✅ NEW COLUMN
                "Extracted Value": " | ".join(label_data[label]["text"]),
                "BBox": merge_bboxes(label_data[label]["bbox"]),
                "Model Confidence": model_conf,   # ✅ NEW COLUMN
            })

        if not labels:
            rows.append({
                "Page": page_label,
                "GT Class": gt_class,
                "Predicted Class": pred_class,
                "Confidence": confidence,
                "Label": "",
                "GT Text": "",
                "Extracted Value": "",
                "BBox": [],
                "Model Confidence": "",
            })
    return rows

def load_gt_text(class_root, doc_name, page_num):

    target_page = f"_page_{page_num}.json"

    # 🔍 loop through all class folders
    for cls in os.listdir(class_root):

        cls_path = os.path.join(class_root, cls)
        zone_output_dir = os.path.join(cls_path, "zone_output")

        if not os.path.isdir(zone_output_dir):
            continue

        for file in os.listdir(zone_output_dir):

            if not file.endswith(target_page):
                continue

            # 🔥 IMPROVED: Normalize both names and use case-insensitive comparison
            file_normalized = normalize_doc_name(file).lower()
            doc_normalized = normalize_doc_name(doc_name).lower()

            # Match if normalized names are equal OR if file starts with doc name
            if file_normalized == doc_normalized or file_normalized.startswith(doc_normalized):

                file_path = os.path.join(zone_output_dir, file)

                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    gt_text_map = {}

                    for label, details in data.items():
                        words = details.get("words", [])
                        text = " ".join([w.get("word", "") for w in words if w.get("word")])
                        gt_text_map[label] = text

                    return gt_text_map
                except Exception as e:
                    print(f"⚠️ Error loading GT text from {file}: {e}")
                    return {}

    print(f"⚠️ GT text not found for: {doc_name} page {page_num}")
    return {}


def normalize_doc_name(name: str):
    """
    Normalize document names to handle various naming conventions.
    Examples:
    - DOC_3.json → DOC_3
    - Doc_3_ver_1_page_5.json → Doc_3
    - Trade Finance_20210520184923_1.00 (2)_page_1.json → Trade Finance_20210520184923_1.00
    - Trade Finance_20210520184923_1.00.json → Trade Finance_20210520184923_1.00
    - Trade Finance_20210520184923_1.00_31Page → Trade Finance_20210520184923_1.00
    - Trade Finance_20210603075228_1.00_21Pages/New document → Trade Finance_20210603075228_1.00
    """
    # Remove .json extension
    name = name.replace(".json", "").strip()
    
    # Remove page references (e.g., "_page_5", "__0")
    name = re.sub(r"(_page_\d+|__\d+).*$", "", name)
    
    # Remove variant suffixes like " (2)", " (3)", etc.
    name = re.sub(r"\s*\(\d+\)\s*$", "", name)
    
    # Remove version suffixes like "_ver_1", "_ver_2"
    name = re.sub(r"_ver_\d+$", "", name)
    
    # Remove page count suffixes like "_31Page", "_52Pages", "_21Pages/New document", "_78Page"
    name = re.sub(r"_\d+\s*Pages?.*$", "", name, flags=re.IGNORECASE)
    
    # Remove debug references like "_debug"
    name = re.sub(r"_debug.*$", "", name)
    
    return name.strip()

def merge_excel_cells(writer, df, sheet_name):

    worksheet = writer.sheets[sheet_name]

    if MODE == "doc":
        merge_cols = ["Page", "GT Class", "Predicted Class", "Confidence"]
        page_col = "Page"
    else:
        merge_cols = ["Doc_Page", "Predicted Class", "Confidence"]
        page_col = "Doc_Page"

    for col in merge_cols:

        if col not in df.columns:
            continue

        col_idx = df.columns.get_loc(col)

        start_row = 1
        prev_value = df.iloc[0][col]
        prev_page = df.iloc[0][page_col]

        for i in range(1, len(df)):

            current_value = df.iloc[i][col]
            current_page = df.iloc[i][page_col]

            # 🔥 KEY FIX: also check page
            if current_value != prev_value or current_page != prev_page:

                if i - start_row > 1:
                    worksheet.merge_range(start_row, col_idx, i, col_idx, prev_value)

                start_row = i + 1
                prev_value = current_value
                prev_page = current_page

        # last block
        if len(df) - start_row > 0:
            worksheet.merge_range(start_row, col_idx, len(df), col_idx, prev_value)
            
            
# ================= MAIN =================
def generate_multi_sheet(json_folder, gt_file, class_root, output_file):

    class_map = load_class_labels(class_root)
    gt_data = load_gt(gt_file)
    
    print(f"📂 Total JSON Files: {len([f for f in os.listdir(JSON_FOLDER) if f.endswith('.json')])}")
    print(f"📂 Total GT Docs: {len(gt_data)}")

    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        all_rows = []   # 🔥 collect everything if class mode

        for file in os.listdir(json_folder):

            if not file.endswith(".json"):
                continue

            raw_doc_name = os.path.splitext(file)[0]
            doc_name = normalize_doc_name(raw_doc_name)

            file_path = os.path.join(json_folder, file)

            with open(file_path, "r") as f:
                data = json.load(f)

            result_json = data.get("result")
            if not result_json:
                result_json = data.get("response", {}).get("result", {})

            rows = process_document(doc_name, result_json, gt_data, class_map)

            if not rows:
                continue

            df = pd.DataFrame(rows)

            # 🔥 ADD DOC NAME COLUMN (important for class mode)
            df["Document"] = doc_name

            df["Page_sort"] = df["Page"].str.extract(r'(\d+)').astype(int)
            df.sort_values(by="Page_sort", inplace=True)
            df.drop(columns=["Page_sort"], inplace=True)

            # ================= MODE SWITCH =================

            def safe_sheet_name(name):
                name = name[:31]                    # limit
                name = name.rstrip(".")            # ❗ remove trailing dot
                name = name.replace("/", "_")      # avoid illegal chars
                name = name.replace("\\", "_")
                return name

            sheet_name = safe_sheet_name(doc_name)

            
            if MODE == "doc":

                df.to_excel(writer, sheet_name=sheet_name, index=False)

                # ✅ always use actual name
                actual_sheet_name = list(writer.sheets.keys())[-1]

                merge_excel_cells(writer, df, actual_sheet_name)

                correct, incorrect, total, accuracy = evaluate_document(df)

                worksheet = writer.sheets[actual_sheet_name]

                # 🔥 start writing after table
                start_row = len(df) + 3   # + header + spacing

                worksheet.write(start_row, 0, "📊 Evaluation")
                worksheet.write(start_row + 1, 0, "Correct")
                worksheet.write(start_row + 1, 1, correct)

                worksheet.write(start_row + 2, 0, "Incorrect")
                worksheet.write(start_row + 2, 1, incorrect)

                worksheet.write(start_row + 3, 0, "Total")
                worksheet.write(start_row + 3, 1, total)

                worksheet.write(start_row + 4, 0, "Accuracy Percentage")
                worksheet.write(start_row + 4, 1, round(accuracy, 4))

                print(f"\n📊 Evaluation for: {actual_sheet_name}")
                print(f"Correct\t\t{correct}")
                print(f"Incorrect\t{incorrect}")
                print(f"Total\t\t{total}")
                print(f"Accuracy Percentage\t{round(accuracy, 4)}")

                print(f"✅ Sheet created: {actual_sheet_name}")
            else:
                all_rows.extend(df.to_dict(orient="records"))
        
        if MODE == "class":

            print("\n📊 Generating class-wise sheets...")

            all_df = pd.DataFrame(all_rows)

            for gt_class in all_df["GT Class"].dropna().unique():

                class_df = all_df[all_df["GT Class"] == gt_class].copy()   # ✅ FIX

                if class_df.empty:
                    continue

                # 🔥 combine doc + page
                class_df["Doc_Page"] = class_df["Document"] + "_" + class_df["Page"]

                # 🔥 drop columns AFTER using them
                class_df.drop(columns=["GT Class", "Document", "Page"], inplace=True)

                # 🔥 reorder
                cols = ["Doc_Page"] + [c for c in class_df.columns if c != "Doc_Page"]
                class_df = class_df[cols]

                sheet_name = str(gt_class)[:31]

                class_df.to_excel(writer, sheet_name=sheet_name, index=False)

                actual_sheet_name = list(writer.sheets.keys())[-1]

                merge_excel_cells(writer, class_df, actual_sheet_name)

                print(f"✅ Class Sheet created: {actual_sheet_name}")

# ================= RUN =================
if __name__ == "__main__":

    JSON_FOLDER = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Test_jsons"
    GT_FILE = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Final Results Sheet.ods"
    CLASS_ROOT = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Test_Data_latest"
    OUTPUT_FILE = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Classes_Analyzing/analysis_output.xlsx"
    
    generate_multi_sheet(JSON_FOLDER, GT_FILE, CLASS_ROOT, OUTPUT_FILE)