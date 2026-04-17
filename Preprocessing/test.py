import base64
import sys
from openai import OpenAI

# --- CONFIGURATION ---
VLLM_API_URL = "http://192.168.170.54:8000/v1"
MODEL_NAME = "/home/ngrental-7/Rupankar_Dev/trade_finance_project/models/Qwen__Qwen3-VL-8B-Thinking"
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
   - Strong Indicators: (1) Label "Packing List" at top as header not indside table or in not description  (2) Itemized table with: Description, Quantity, Net Weight, Gross Weight, Volume (CBM/M3), Marks & Numbers, Package/Bag Count, (3) Shipment Totals, (4) Carton/Container counts, (5) Packaging breakdown, (6) Measurement units (PCS, CTN, KGS).
   - Negative Indicator: Reject as PL if the header "Commercial Invoice" is present.
   - Decision Logic: Count ≥ 3 indicators -> "Packing_List".

4. Commercial Invoice (CI)
   - Scope: Financial sale transaction document, rarely multi-page.
   - Strong Indicators: (1) Label "Commercial Invoice" or "Invoice" at top as header not indside table or in not description (2) Pricing table: Currency, Amount, Total, Unit Price, Quantities, (3) Identifiers: Invoice Number, Date, Proforma Ref, (4) Seller/Buyer info: Exporter/Beneficiary, Importer/Applicant, Consignee, Bill To, Notify Party, (5) PO/Purchase Order/Proforma Number.
   - Decision Logic: Count ≥ 4 indicators -> "Commercial_Invoice".


5. Bill of Exchange (BoE)
   - Scope: Financial instrument/Draft, single page.
   - Strong Indicators: (1) Label "Bill of Exchange", "Draft", "Second Unpaid", or "First Bill", (2) Payment language: "Pay to the order of", "Pay against this bill", "Pay the sum of", (3) Parties: Drawer, Drawee, Payee, (4) Amount in figures AND words, (5) Payment Term/Tenor (At Sight, X days from BL/Invoice date), (6) "For and On Behalf of" text, (7) Stamp/Revenue stamp, (8) Maturity date.
   - Decision Logic: Count ≥ 4 indicators -> "Bill_of_Exchange".

6. Covering Schedule (CS)
   - Scope: Bank/Exporter instruction letter, multi-page.
   - Strong Indicators: (1) Header: "Bill covering schedule", "Collection schedule", "Covering letter", or "Export documentary covering schedule", (2) Issuing/Receiving Bank names, (3) Bank name in body, (4) LC Summary: Exporter, Importer, Tenor, LC Number, Bill Amount,Documentary credit Number  (5) Bill/Our reference number, (6) Document Checklist table (Originals/Copies), (7) Presentation statements: "We enclose documents", "Please remit proceeds", (8) Compliance: UCP 600 or ICC rules, (9) Banking footer/SWIFT code. 
   - Decision Logic: Count ≥ 5 indicators -> "Covering_Schedule".

7. others
   - If the document does not reach the minimum count threshold for any of the 6 classes above, classify as "others".
   

Instrucitons:

1. If a document matches 80% or more of the required indicator count for any class, and the page number is 3 or 4, then do not classify it as "others".
Instead, classify it as the corresponding class and append "(low confidence)" to the class name (e.g., Class_Name (low confidence)).


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

def get_raw_model_response(image_path):
    client = OpenAI(base_url=VLLM_API_URL, api_key="sk-vllm")
    
    print(f"Encoding image: {image_path}")
    img_64 = encode_img_b64(image_path)
    
    print(f"Calling API for raw response...")
    try:
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
        
        raw_content = response.choices[0].message.content
        
        print("\n" + "="*20 + " RAW MODEL OUTPUT START " + "="*20)
        print(raw_content)
        print("="*21 + " RAW MODEL OUTPUT END " + "="*21 + "\n")
        
    except Exception as e:
        print(f"API Error: {e}")

if __name__ == "__main__":
    # Choose one of the images that produced the "No JSON found" error
    TEST_IMAGE = "/datadrive2/IDF_AL_MASRAF/Al_MASHRAF_523_categorized_data/BOL/Trade Finance_20210216165325_1.00_page_14.jpeg"
    
    
    get_raw_model_response(TEST_IMAGE)