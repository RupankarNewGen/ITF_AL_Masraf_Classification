import matplotlib.pyplot as plt
import matplotlib.patches as patches

# JSON structure
json_data = {
    "form": [
        {
            "id": 0,
            "box": [623, 664, 792, 686],
            "label": "other",
            "text": "Country of Origin:",
            "words": [
                {"text": "Country", "box": [624, 666, 697, 684]},
                {"text": "of", "box": [704, 665, 723, 681]},
                {"text": "Origin:", "box": [729, 665, 791, 683]},
            ],
            "linking": [
                [0, 2]  # Corrected linking between token indices 0 and 2
            ]
        },
        # Add more form sections as needed
    ]
}

# Extract tokens and linking from JSON
tokens = json_data["form"][0]["words"]
linking = json_data["form"][0]["linking"]

# Create a new figure
fig, ax = plt.subplots()

# Draw tokens and linking
for token in tokens:
    text = token["text"]
    box = token["box"]
    ax.add_patch(patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False))
    ax.text((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, text, ha="center", va="center")

for link_pair in linking:
    from_token_idx, to_token_idx = link_pair
    from_box = tokens[from_token_idx]["box"]
    to_box = tokens[to_token_idx]["box"]
    ax.plot([from_box[2], to_box[0]], [(from_box[1] + from_box[3]) / 2, (to_box[1] + to_box[3]) / 2], color="blue")

# Set plot limits
ax.set_xlim(600, 800)
ax.set_ylim(600, 700)

# Show the plot
plt.show()
