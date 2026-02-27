import json
import re

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

if __name__ == "__main__":
    # PASTE YOUR RAW RESPONSE HERE
    RAW_RESPONSE = """
==================== RAW MODEL OUTPUT START ====================
So, let's analyze this document step by step. First, the document is labeled "PACKING LIST" at the top, which is a strong indicator for Packing List (PL). Now, let's check the strong indicators for PL:

1. Label "Packing List" at top: Yes, it has "PACKING LIST" and the Chinese "裝箱單".
2. Itemized table with Description, Quantity, Net Weight, Gross Weight, Volume: The table has columns for MARKS & NO., DESCRIPTION, QUANTITY, N.W. (KGS), G.W. (KGS), MEAS. (CBM). So that's present.
3. Shipment Totals: The TOTAL line shows "TOTAL: (72 PALLETS=4694 CARTONS)" and "TOTAL: FOUR THOUSAND SIX HUNDRED AND NINETY FOUR (4694) BATTERY ONLY." So shipment totals are there.
4. Carton/Container counts: The total mentions 72 pallets and 4694 cartons.
5. Packaging breakdown: The itemized list shows different battery types with quantities, so packaging breakdown is present.
6. Measurement units (PCS, CTN, KGS): The table uses KGS for weight and CBM for volume, and the quantity is in "BATTERY" (which is a unit here).

Now, check for negative indicator: The header "Commercial Invoice" is present? Wait, the document has "INVOICE NO.: LBPV214464021" but the main label is "PACKING LIST". The negative indicator for PL is if "Commercial Invoice" header is present. Here, the header is "PACKING LIST", so no negative indicator.

Now count the indicators for PL:

1. Label: Yes (1)
2. Itemized table: Yes (2)
3. Shipment Totals: Yes (3)
4. Carton/Container counts: Yes (4)
5. Packaging breakdown: Yes (5)
6. Measurement units: Yes (6)

Wait, the PL decision logic is count ≥ 3 indicators. Let's confirm each:

- Label "Packing List" at top: Yes (1)
- Itemized table with Description, Quantity, Net Weight, Gross Weight, Volume: Yes (2)
- Shipment Totals: Yes (3)
- Carton/Container counts: Yes (4)
- Packaging breakdown: Yes (5)
- Measurement units: Yes (6)

So that's 6 indicators, which is ≥3. Now, check if any other categories might apply. For example, Commercial Invoice (CI) has a label "Commercial Invoice" or "Invoice" at top. Here, the top is "PACKING LIST", so CI is out. Bill of Lading (BL) has labels like "Bill of Lading", which isn't present. Certificate of Origin (COO) has "Certificate of Origin" label, not here. Bill of Exchange (BoE) has specific labels, not present. Covering Schedule (CS) has headers like "Bill covering schedule", not here.

Also, check if the document is a PL. The negative indicator for PL is if "Commercial Invoice" header is present. The document has "INVOICE NO." but the main header is "PACKING LIST", so no negative indicator.

So the count for PL is 6 indicators, which meets the threshold (≥3). Therefore, classification is Packing_List.
</think>

{
  "classification": "Packing_List",
  "reasoning": "Detected 6 strong indicators for Packing List (label, itemized table, shipment totals, carton counts, packaging breakdown, measurement units) meeting the ≥3 threshold"
}
"""

    # Execute extraction
    result = extract_json_from_raw_text(RAW_RESPONSE)

    # Print Result
    print("--- EXTRACTED JSON ---")
    print(json.dumps(result, indent=4))
    print("----------------------")