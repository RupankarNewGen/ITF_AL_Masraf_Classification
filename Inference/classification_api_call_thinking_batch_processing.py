import os
import json
import base64
import re
import sys
import time
from tqdm import tqdm
from openai import OpenAI

# --- CONFIGURATION ---
VLLM_API_URL = "http://192.168.170.76:8000/v1"
MODEL_NAME = "/home/ng6309/datascience/hridesh/Qwen__Qwen3-VL-8B-Thinking"

WINDOW_SIZE = 4  # Number of pages per request
OVERLAP = 1     # Contextual overlap

SYSTEM_PROMPT_CONTEXTUAL = """You are an expert Trade Finance Document Auditor. Analyze the provided image to classify it into one of 7 categories based on the specific scoring thresholds provided.

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

ADDITIONAL CONTEXTUAL INSTRUCTION:
I am providing a sequence of images from the same folder. Additionally, check the previous pages as context to predict the class and reasoning for the current page.

### OUTPUT REQUIREMENT:
Return a single JSON object containing an array called "pages" with the prediction for every image provided in sequence.
{
  "pages": [
    {
      "page_number": 1,
      "file_name": "name",
      "classification": "Class_Name_Only",
      "reasoning": "One-line explanation"
    }
  ]
}
"""

def encode_img_b64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def extract_json_from_thinking(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"pages": []}
    except Exception:
        return {"pages": []}

def process_all_folders(manifest_path, output_dir):
    client = OpenAI(base_url=VLLM_API_URL, api_key="sk-vllm")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        folder_groups = json.load(f)

    # Folders are now top-level keys in your nested JSON
    folder_ids = list(folder_groups.keys())
    print(f"Total folders found in manifest: {len(folder_ids)}")

    for folder_id in tqdm(folder_ids, desc="Processing Folders"):
        img_paths = folder_groups[folder_id]
        save_path = os.path.join(output_dir, f"{folder_id}.json")

        # Skip if folder already fully processed
        if os.path.exists(save_path):
            continue

        final_pages_results = []
        step = WINDOW_SIZE - OVERLAP
        
        for i in range(0, len(img_paths), step):
            batch = img_paths[i : i + WINDOW_SIZE]
            if i > 0 and len(batch) <= OVERLAP: 
                break
            
            user_content = [{"type": "text", "text": f"Analyze this sequence for folder {folder_id}:"}]
            for path in batch:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_img_b64(path)}"}
                })

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT_CONTEXTUAL},
                              {"role": "user", "content": user_content}],
                    temperature=0.0
                )
                
                batch_data = extract_json_from_thinking(response.choices[0].message.content)
                pages_list = batch_data.get("pages", [])

                if i > 0:
                    final_pages_results.extend(pages_list[OVERLAP:])
                else:
                    final_pages_results.extend(pages_list)

            except Exception as e:
                print(f"\nError in batch for {folder_id} at index {i}: {e}", flush=True)

        # Save single output JSON with the folder name
        final_json = {
            "folder_id": folder_id,
            "total_pages": len(final_pages_results),
            "pages": final_pages_results
        }
        
        with open(save_path, 'w', encoding='utf-8') as out_f:
            json.dump(final_json, out_f, indent=4)
        
        sys.stdout.flush()

if __name__ == "__main__":
    # Path to your NEW nested JSON manifest
    MANIFEST = "targeted_sibling_expansion_nested.json"
    # Output directory for individual folder JSONs
    OUTPUT = "/datadrive2/IDF_AL_MASRAF/CONTEXT_BATCH_RESULTS_NESTED"
    
    process_all_folders(MANIFEST, OUTPUT)