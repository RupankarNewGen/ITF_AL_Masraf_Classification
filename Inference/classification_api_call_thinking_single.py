import os
import json
import base64
import re
import time
import io
import sys
from tqdm import tqdm
from openai import OpenAI
from PIL import Image

# --- CONFIGURATION ---
VLLM_API_URL = "http://192.168.170.54:8000/v1"
MODEL_NAME = "/home/ngrental-7/Rupankar_Dev/trade_finance_project/models/Qwen__Qwen3-VL-8B-Thinking"
TARGET_DPI = None  
TARGET_SIZE = (896,1152) # Change this to (width,height) to enable resizing, or leave as None

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

def extract_json_from_raw_text(text):
    """
    Finds the JSON block only AFTER the </think> tag to avoid 
    parsing internal thoughts.
    """
    try:
        # 1. Look for the end of the thinking block
        if "</think>" in text:
            # Take only the content after the closing think tag
            text_after_think = text.split("</think>")[-1].strip()
        else:
            # Fallback if the model didn't use tags or they were stripped
            text_after_think = text.strip()

        # 2. Extract JSON from the remaining text
        # re.DOTALL ensures it matches across multiple lines
        json_match = re.search(r'\{.*\}', text_after_think, re.DOTALL)
        
        if json_match:
            return json.loads(json_match.group())
            
        return {"classification": "error", "reasoning": "No JSON found below think tag"}
    except Exception as e:
        return {"classification": "error", "reasoning": f"Parse Error: {str(e)}"}

def encode_img_b64(image_path, target_dpi=None, target_size=None):
    with Image.open(image_path) as img:
        print(f"Original image size: {img.size}") # Debug: Print original image info
        
        if target_dpi is not None:
            current_dpi = img.info.get('dpi', (300, 300))[0]
            scale_factor = target_dpi / current_dpi
            if scale_factor != 1.0:
                new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # New Resize logic
        if target_size is not None:
            print(f"Resizing image to: {target_size}")
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        else:
            print("No target size provided. Not resizing.")
        
        print(f"Final image size for encoding: {img.size}, DPI: {target_dpi if target_dpi else 'Original'}")  # Debug
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
       

def call_qwen_api(image_path, target_dpi, target_size):
    client = OpenAI(base_url=VLLM_API_URL, api_key="sk-vllm")
    try:
        img_64 = encode_img_b64(image_path, target_dpi=target_dpi, target_size=target_size)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_CLASSIFY},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_64}"}}
                ]}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return json.dumps({"classification": "error", "reasoning": str(e)})

def process_manifest(manifest_path, output_folder, target_dpi, target_size):
    os.makedirs(output_folder, exist_ok=True)
    with open(manifest_path, 'r', encoding='utf-8') as f:
        image_paths = json.load(f)

    print(f"Loaded {len(image_paths)} images.", flush=True)
    total_start_time = time.perf_counter()
    processing_times = []
    
    for img_path in tqdm(image_paths, desc="Processing"):
        filename = os.path.basename(img_path)
        name_only = os.path.splitext(filename)[0]
        save_path = os.path.join(output_folder, f"{name_only}.json")

        should_process = True
        
        if os.path.exists(save_path):
            try:
                with open(save_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if existing_data.get("classification") != "error":
                        should_process = False
            except Exception:
                should_process = True

        if should_process:
            img_start = time.perf_counter()
            
            raw_output = call_qwen_api(img_path, target_dpi, target_size)
            clean_json = extract_json_from_raw_text(raw_output)
            
            with open(save_path, "w", encoding="utf-8") as out_f:
                json.dump(clean_json, out_f, indent=4)
            
            img_end = time.perf_counter()
            duration = img_end - img_start
            processing_times.append(duration)
            print(f" -> {name_only} processed in {duration:.2f}s", flush=True)
            sys.stdout.flush()

    total_end_time = time.perf_counter()
    if processing_times:
        print(f"\nOVERALL TIME: {total_end_time - total_start_time:.2f}s", flush=True)

if __name__ == "__main__":
    MANIFEST = "/datadrive2/IDF_AL_MASRAF/Import_LC_Acceptance_manifest.json"
    OUTPUT = "/datadrive2/IDF_AL_MASRAF/LC_Acceptence_Full_Result"
    process_manifest(MANIFEST, OUTPUT, target_dpi=TARGET_DPI, target_size=TARGET_SIZE)