import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

def setup_parallel_model(net, device_ids=None):
    """
    Setup model for parallel processing.
    
    Args:
        net: The model to parallelize
        device_ids: List of GPU devices to use. If None, use all available GPUs.
    Returns:
        Parallelized model
    """
    if torch.cuda.device_count() > 1:
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        print(f"Using {len(device_ids)} GPUs: {device_ids}")
        return DataParallel(net, device_ids=device_ids)
    return net

def prepare_batch_data(input_data, device):
    """
    Prepare input data for batch processing.
    
    Args:
        input_data: Dictionary of input tensors
        device: Target device
    Returns:
        Processed input data
    """
    processed_data = {}
    for key, value in input_data.items():
        try:
            # Add batch dimension if not present
            if not isinstance(value, torch.Tensor):
                continue
            if len(value.shape) == 0:
                continue
            if len(value.shape) == 3:  # Assuming image data
                value = torch.unsqueeze(value, 0)
            processed_data[key] = value.to(device)
        except Exception as e:
            print(f"Warning: Could not process key {key}: {str(e)}")
            processed_data[key] = value
    return processed_data


def parallel_inference(net, input_data, batch_size=1):
    """
    Run inference in parallel.
    
    Args:
        net: The model (will be parallelized if multiple GPUs available)
        input_data: Input data dictionary
        batch_size: Batch size for processing
    Returns:
        Model outputs and loss dictionary
    """
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Parallelize model if possible
    if torch.cuda.is_available():
        net = setup_parallel_model(net)
    net = net.to(device)
    net.eval()
    
    # Print device information
    current_device = next(net.parameters()).device
    print(f"Model is running on: {current_device}")
    print(f"Number of GPUs being used: {torch.cuda.device_count()}")
    
    # Prepare input data
    processed_data = prepare_batch_data(input_data, device)
    
    # Run inference
    with torch.no_grad():
        try:
            head_outputs, loss_dict = net(processed_data)
            return head_outputs, loss_dict
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            raise

# Example usage
def run_inference_pipeline(net, image, json_obj, tokenizer, backbone_type):
    """
    Complete inference pipeline with parallel processing.
    """
    # Get input data
    input_data = getitem_geo(image, json_obj, tokenizer, backbone_type)
    
    # Run parallel inference
    try:
        head_outputs, loss_dict = parallel_inference(net, input_data)
        return head_outputs, loss_dict
    except Exception as e:
        print(f"Pipeline error: {str(e)}")
        raise
    
    
# Simple usage
# head_outputs, loss_dict = parallel_inference(net, input_data)

# # Or complete pipeline
# results = run_inference_pipeline(net, image, json_obj, tokenizer, backbone_type)

# # Specify which GPUs to use (e.g., GPUs 0 and 1)
# device_ids = [0, 1]
# net = setup_parallel_model(net, device_ids=device_ids)