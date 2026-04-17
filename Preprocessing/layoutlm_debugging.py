import json
import numpy as np
from PIL import Image, ImageDraw
from transformers import LayoutLMTokenizer


# -----------------------------
# SCALE BOUNDING BOX
# -----------------------------
def scale_bounding_box(box, width_scale, height_scale):

    x0, y0, x1, y1 = box

    # ensure correct ordering
    x0, x1 = sorted([x0, x1])
    y0, y1 = sorted([y0, y1])

    scaled_box = [
        int(x0 * width_scale),
        int(y0 * height_scale),
        int(x1 * width_scale),
        int(y1 * height_scale)
    ]

    # clamp to LayoutLM range
    scaled_box = [max(0, min(v, 999)) for v in scaled_box]

    return scaled_box


# -----------------------------
# NORMALIZE OCR RESULT
# -----------------------------
def normalize_ocr_result(image, ocr_result):

    words = []
    bbox = []

    width, height = image.size

    width_scale = 1000 / width
    height_scale = 1000 / height

    if isinstance(ocr_result, dict):
        ocr_result = ocr_result["ocrContent"]

    for ll in ocr_result:

        word = ll["word"].strip()

        if word == "":
            continue

        box = [ll["x1"], ll["y1"], ll["x2"], ll["y2"]]

        words.append(word)
        bbox.append(box)

    normalized_boxes = []

    for box in bbox:

        scaled_box = scale_bounding_box(box, width_scale, height_scale)
        normalized_boxes.append(scaled_box)

    assert len(words) == len(normalized_boxes), "Words and boxes mismatch"

    return words, normalized_boxes


# -----------------------------
# DEBUG FUNCTION
# -----------------------------
def debug_layoutlm_input(image, words, boxes, tokenizer):

    print("\n========== LAYOUTLM INPUT DEBUG ==========\n")

    # ------------------------
    # WORD vs BOX CHECK
    # ------------------------
    print("1️⃣ Word–BBox Count")
    print("Words:", len(words))
    print("Boxes:", len(boxes))

    if len(words) != len(boxes):
        print("❌ Mismatch detected")

    print()

    # ------------------------
    # EMPTY WORD CHECK
    # ------------------------
    print("2️⃣ Empty Words")

    empty_words = [i for i,w in enumerate(words) if w.strip()==""]
    print("Empty words:", len(empty_words))

    print()

    # ------------------------
    # BBOX VALIDITY CHECK
    # ------------------------
    print("3️⃣ Bounding Box Validity")

    bad_boxes = []
    zero_area = []

    for i,box in enumerate(boxes):

        x0,y0,x1,y1 = box

        if not (0 <= x0 <= 1000 and 0 <= y0 <= 1000 and 0 <= x1 <= 1000 and 0 <= y1 <= 1000):
            bad_boxes.append((i,box))

        if x1 <= x0 or y1 <= y0:
            zero_area.append((i,box))

    print("Invalid range boxes:", len(bad_boxes))
    print("Zero area boxes:", len(zero_area))

    if bad_boxes[:5]:
        print("Examples:", bad_boxes[:5])

    print()

    # ------------------------
    # BBOX DISTRIBUTION
    # ------------------------
    print("4️⃣ Bounding Box Distribution")

    boxes_np = np.array(boxes)

    print("Min:", boxes_np.min(axis=0))
    print("Max:", boxes_np.max(axis=0))

    print()

    # ------------------------
    # TOKENIZATION CHECK
    # ------------------------
    print("5️⃣ Tokenization Alignment")

    encoding = tokenizer(
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=512
    )

    input_ids = encoding["input_ids"]
    token_boxes = encoding["bbox"]

    print("Tokens:", len(input_ids))
    print("Token boxes:", len(token_boxes))

    if len(input_ids) != len(token_boxes):
        print("❌ Token–BBox mismatch")
    else:
        print("✅ Token–BBox aligned")

    print()

    # ------------------------
    # SAMPLE TOKENS
    # ------------------------
    print("6️⃣ Sample Tokens")

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    for i in range(20):
        print(tokens[i], token_boxes[i])

    print()

    # ------------------------
    # VISUALIZE BOXES
    # ------------------------
    print("7️⃣ Visualizing bounding boxes")

    img = image.copy()
    draw = ImageDraw.Draw(img)

    for box in boxes[:50]:

        x0 = box[0] * image.width / 1000
        y0 = box[1] * image.height / 1000
        x1 = box[2] * image.width / 1000
        y1 = box[3] * image.height / 1000

        draw.rectangle([x0,y0,x1,y1], outline="red", width=2)

    img.show()

    print("\n========== DEBUG COMPLETE ==========\n")


# -----------------------------
# MAIN SCRIPT
# -----------------------------
if __name__ == "__main__":

    image_path = "/datadrive2/IDF_AL_MASRAF/DocumentDumpImages/TF2028500009/Import LC Settlement/TF202850000902/A09862_20210125163753_1.00_page_3.jpeg"
    ocr_json_path = "/datadrive2/IDF_AL_MASRAF/DocumentDumpImages/TF2028500009/Import LC Settlement/TF202850000902/A09862_20210125163753_1.00_page_3.jpeg"

    image = Image.open(image_path).convert("RGB")

    with open(ocr_json_path) as f:
        ocr_result = json.load(f)

    tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

    words, boxes = normalize_ocr_result(image, ocr_result)

    debug_layoutlm_input(image, words, boxes, tokenizer)