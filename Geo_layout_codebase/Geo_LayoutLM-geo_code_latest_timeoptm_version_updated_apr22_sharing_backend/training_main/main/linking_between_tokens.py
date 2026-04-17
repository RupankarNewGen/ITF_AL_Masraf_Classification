import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
# Load the document image
document_image_path = "/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Layoutlmv3_code_base/PL_complete_data/data_in_funsd_format/dataset/custom_geo/testing_data/images/Packing_List_57_page_2.png"
json_path= '/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Layoutlmv3_code_base/PL_complete_data/data_in_funsd_format/dataset/custom_geo/testing_data/annotations/Packing_List_57_page_2.json'
document_image = Image.open(document_image_path)
with open(json_path, "r") as json_file:
    json_data = json.load(json_file)

for section in json_data["form"]:
    tokens = section["words"]
    linking = section["linking"]

    # Create a new figure for each section
    fig, ax = plt.subplots()

    # Display the document image
    ax.imshow(document_image)

    # Draw tokens on the image
    for token in tokens:
        text = token["text"]
        box = token["box"]
        ax.add_patch(patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor="red"))
        ax.text((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, text, ha="center", va="center", color="red")

    # Accumulate linking lines for the section
    linking_lines = []
    for link_pair in linking:
        from_token_idx, to_token_idx = link_pair
        
        # Make sure the indices are within the range of available tokens
        if 0 <= from_token_idx < len(tokens) and 0 <= to_token_idx < len(tokens):
            from_box = tokens[from_token_idx]["box"]
            to_box = tokens[to_token_idx]["box"]
            linking_lines.append(([from_box[2], to_box[0]], [(from_box[1] + from_box[3]) / 2, (to_box[1] + to_box[3]) / 2]))

    # Draw all accumulated linking lines
    for line_coords in linking_lines:
        ax.plot(line_coords[0], line_coords[1], color="blue")

    # Show the overlaid image for the current section
    plt.show()

