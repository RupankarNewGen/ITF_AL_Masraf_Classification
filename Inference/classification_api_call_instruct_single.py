import os
import json
import base64
import time
from tqdm import tqdm
from openai import OpenAI

VLLM_API_URL = "http://192.168.170.76:8000/v1"
MODEL_NAME = "/home/ng6309/datascience/hridesh/Qwen__Qwen3-VL-8B-Instruct"

SYSTEM_PROMPT_CLASSIFY = """You are an expert Trade Finance Document Auditor. Analyze the provided image to classify it into one of 7 categories based on the specific scoring thresholds provided.

### CATEGORY SPECIFICATIONS & SCORING RULES:

1. Bill of Lading (BL)
   - Scope: Sea shipment document, single or multi-page.
   - Strong Indicators: (1) Label "Bill of Lading" at top, (2) Carrier Name, (3) Shipping Line, (4) Shipper Name, (5) B/L Number (B/L No., BL No.), (6) Place of Receipt, (7) Port of Loading, (8) Port of Discharge, (9) Place of Delivery, (10) Vessel Name, (11) Voyage Number, (12) Container Number, (13) Seal Number, (14) Freight terms (Prepaid/Collect/Ocean Freight).
   - Decision Logic: Count ≥ 8 indicators -> "Bill_of_Lading".

2. Certificate of Origin (COO)
   - Scope: International trade origin cert, usually single page.
   - Strong Indicators: (1) Label "Certificate of Origin" at top, (2) Issued by Authorized Body (Chamber of Commerce/Ministry/Export Promotion Council), (3) Certificate Number with Port info, (4) Exporter Name, (5) Exporter Address, (6) "Issued in [Country]" statement, (7) Exporter Declaration, (8) Goods Consigned From, (9) Goods Consigned To, (10) Origin Criteria.
   - Decision Logic: Count ≥ 6 indicators -> "Certificate_of_Origin".

3. Packing List (PL)
   - Scope: Shipping document describing contents, multi-page (Page 1/3, etc.).
   - Strong Indicators: (1) Label "Packing List" at top, (2) Itemized table with: Description, Quantity, Net Weight, Gross Weight, Volume (CBM/M3), Marks & Numbers, Package/Bag Count, (3) Shipment Totals, (4) Carton/Container counts, (5) Packaging breakdown, (6) Measurement units (PCS, CTN, KGS).
   - Negative Indicator: Reject as PL if the header "Commercial Invoice" is present.
   - Decision Logic: Count ≥ 3 indicators -> "Packing_List".

4. Commercial Invoice (CI)
   - Scope: Financial sale transaction document, rarely multi-page.
   - Strong Indicators: (1) Label "Commercial Invoice" or "Invoice" at top, (2) Pricing table: Currency, Amount, Total, Unit Price, Quantities, (3) Identifiers: Invoice Number, Date, Proforma Ref, (4) Seller/Buyer info: Exporter/Beneficiary, Importer/Applicant, Consignee, Bill To, Notify Party, (5) PO/Purchase Order/Proforma Number.
   - Decision Logic: Count ≥ 4 indicators -> "Commercial_Invoice".

5. Bill of Exchange (BoE)
   - Scope: Financial instrument/Draft, single page.
   - Strong Indicators: (1) Label "Bill of Exchange", "Draft", "Second Unpaid", or "First Bill", (2) Payment language: "Pay to the order of", "Pay against this bill", "Pay the sum of", (3) Parties: Drawer, Drawee, Payee, (4) Amount in figures AND words, (5) Payment Term/Tenor (At Sight, X days from BL/Invoice date), (6) "For and On Behalf of" text, (7) Stamp/Revenue stamp, (8) Maturity date.
   - Decision Logic: Count ≥ 4 indicators -> "Bill_of_Exchange".

6. Covering Schedule (CS)
   - Scope: Bank/Exporter instruction letter, multi-page.
   - Strong Indicators: (1) Header: "Bill covering schedule", "Collection schedule", "Covering letter", or "Export documentary covering schedule", (2) Issuing/Receiving Bank names, (3) Bank name in body, (4) LC Summary: Exporter, Importer, Tenor, LC Number, Bill Amount, (5) Bill/Our reference number, (6) Document Checklist table (Originals/Copies), (7) Presentation statements: "We enclose documents", "Please remit proceeds", (8) Compliance: UCP 600 or ICC rules, (9) Banking footer/SWIFT code.
   - Decision Logic: Count ≥ 5 indicators -> "Covering_Schedule".

7. others
   - If the document does not reach the minimum count threshold for any of the 6 classes above, classify as "others".

### OUTPUT REQUIREMENT:
Return ONLY a JSON object with the classification result and a one-line reasoning explaining the indicator count.
{
  "classification": "Class_Name_Only",
  "reasoning": "One-line explanation of detected indicators and scoring"
}
"""

def encode_img_b64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def call_qwen_classification(image_path):
    client = OpenAI(base_url=VLLM_API_URL, api_key="sk-vllm")
    try:
        img_64 = encode_img_b64(image_path)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_CLASSIFY},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_64}"}}
                ]}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return json.dumps({"classification": "error", "message": str(e)})

def process_manifests(manifest_paths, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    all_images = []
    for m_path in manifest_paths:
        with open(m_path, 'r') as f:
            all_images.extend(json.load(f))
    
    total_images = len(all_images)
    print(f"Total images in manifest: {total_images}")
    
    processed_this_run = 0
    skipped_count = 0
    start_time = time.time()

    for img_path in tqdm(all_images, desc="Classifying"):
        filename = os.path.basename(img_path)
        name_only = os.path.splitext(filename)[0]
        save_path = os.path.join(output_folder, f"{name_only}.json")

        # CHECK: If file exists, skip it
        if not os.path.exists(save_path):
            result_string = call_qwen_classification(img_path)
            with open(save_path, "w", encoding="utf-8") as out_f:
                out_f.write(result_string)
            processed_this_run += 1
        else:
            skipped_count += 1

    end_time = time.time()
    
    total_duration = end_time - start_time
    # Avg time is calculated only based on images actually processed this run
    avg_duration = total_duration / processed_this_run if processed_this_run > 0 else 0

    print("\n" + "="*50)
    print(f"PROCESS COMPLETED")
    print(f"Total Images in Manifest: {total_images}")
    print(f"Images Skipped (Existing): {skipped_count}")
    print(f"Images Processed Now:     {processed_this_run}")
    print(f"Total Time for Run:       {total_duration:.2f} seconds")
    if processed_this_run > 0:
        print(f"Avg Time per Image:       {avg_duration:.2f} seconds")
    print("="*50)

if __name__ == "__main__":
    INPUT_MANIFESTS = ["/datadrive2/IDF_AL_MASRAF/targeted_sibling_expansion.json"]
    OUTPUT_DIR = "/datadrive2/IDF_AL_MASRAF/QWEN_CLASSIFICATION_RESULTS_500_TARGETED_SIBLING_EXPANSION"
    process_manifests(INPUT_MANIFESTS, OUTPUT_DIR)