

import os
import torch


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