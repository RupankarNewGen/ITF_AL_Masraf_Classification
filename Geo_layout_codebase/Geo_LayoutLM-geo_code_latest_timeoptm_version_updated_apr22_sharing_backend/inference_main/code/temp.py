

import torch
import os
import torch
import os


def save_loaded_model(net, save_path, filename="model_eval.pth"):
    """
    Save an already loaded model in eval mode.
    
    Args:
        net: The loaded PyTorch model
        save_path: Directory to save the model
        filename: Name of the saved model file
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Ensure model is in eval mode
    net.eval()
    
    # Full path for saving
    full_save_path = os.path.join(save_path, filename)
    
    # Create save dictionary with model state and metadata
    save_dict = {
        "state_dict": net.state_dict(),
        "eval_mode": True
    }
    
    # Save the model
    torch.save(save_dict, full_save_path)
    print(f"Model saved in eval mode at: {full_save_path}")

# Example usage with your existing code:
def process_and_save_model(geo_cfg, device, model_path, save_path):
    """
    Load model and save it in eval mode
    """
    # Your existing loading code
    net = get_model(geo_cfg)
    load_model_weight(net, device)
    net.to(device)
    net.eval()
    
    # Save the loaded model
    save_loaded_model(net, save_path)
    
    return net


import torch

def load_saved_eval_model(net, model_path, device):
    """
    Load a model that was saved in eval mode.
    
    Args:
        net: Initial model instance
        model_path: Path to the saved model
        device: Device to load the model on
    Returns:
        Loaded model in eval mode
    """
    print(f"Loading model from: {model_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get state dict based on save format
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
        
    # Process state dict keys (keeping your original logic)
    new_state_dict = {}
    valid_keys = net.state_dict().keys()
    invalid_keys = []
    
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("net."):
            new_k = new_k[len("net."):]
        
        if new_k in valid_keys:
            new_state_dict[new_k] = v
        else:
            invalid_keys.append(new_k)
    
    if invalid_keys:
        print(f"These keys are invalid in the checkpoint: [{','.join(invalid_keys)}]")
    
    # Load the processed state dict
    net.load_state_dict(new_state_dict)
    
    # Move to device and set eval mode
    net = net.to(device)
    net.eval()
    
    print("Model loaded successfully in eval mode")
    return net

# Example usage with complete pipeline
def get_inference_model(geo_cfg, device, model_path):
    """
    Complete pipeline to load saved model for inference
    """
    # Initialize model
    net = get_model(geo_cfg)
    
    # Load the saved weights
    net = load_saved_eval_model(net, model_path, device)
    
    return net

exit('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')


import os
import shutil
from fuzzywuzzy import fuzz

def get_eval_kwargs_geolayoutlm_vie(geo_clsses_path):
    print(geo_clsses_path)
    class_names = get_class_names(geo_clsses_path)
    bio_class_names = ["O"]
    for class_name in class_names:
        if not class_name.startswith('O'):
            bio_class_names.extend([f"B-{class_name}", f"I-{class_name}"])
    eval_kwargs = {
        "bio_class_names": bio_class_names,
    }
    return eval_kwargs

def get_class_names(dataset_root_path):
    class_names_file = os.path.join(dataset_root_path)#, "class_names.txt")
    class_names = (
        open(class_names_file, "r", encoding="utf-8").read().strip().split("\n")
    )
    print(class_names)
    return class_names

ot = get_eval_kwargs_geolayoutlm_vie('class_names_temp.txt')
print(ot)
exit('OK')



from fuzzywuzzy import fuzz

str1 = "a b a c"
str2 = "a b c"

similarity_score = fuzz.token_sort_ratio(str1, str2)
print(similarity_score)  # This should also output a high score, typically 100, as it treats them as the same ordered set.




import torch
import torch.nn.functional as F

# Example logits for classification
logits = torch.tensor([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])

# Apply softmax along the last dimension (-1)
probs = F.softmax(logits, dim=-1)

print("Logits:", logits)
print("Probabilities:", probs)

# output a high score (e.g., 100), indicating a strong match.
exit('OK')



def clean_text(text):
    # Remove leading and trailing '/', '-', or ':'
    token_to_strip = "/-:._'^"
    return text.strip(token_to_strip)

value1 = 'V NDIA ENGINEERING ANDCONSTRUCTION PVT LTD wr 11 .I000-'
value1 = 'to , grasim industries ltd binaga , karwar 581307 . -/'
value2 = 'v ndia engineering andconstruction pvt ltd'
value2 = 'To, GRASIM INDUSTRIES LTD BINAGA, KARWAR - 581307.'
value1 = value1.replace(" ", "")
value2 = value2.replace(" ", "")
# if field_name in ['purchase_order_number', 'pan_number', 'vendor_name']:
value1 = clean_text(value1)
value2 = clean_text(value2)
print('value1', value1)
print('value2', value2)
print(fuzz.ratio(value1, value2))
print(fuzz.ratio(value2.lower(), value1.lower()))
print(fuzz.ratio(value1.lower(), value2.lower()))
exit('OK')


    