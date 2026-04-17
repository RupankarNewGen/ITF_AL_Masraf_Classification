import torch
import multiprocessing
import psutil
import os

def get_cpu_info():
    """
    Get detailed information about available CPU resources.
    
    Returns:
        dict: Dictionary containing CPU information
    """
    cpu_info = {
        'physical_cores': psutil.cpu_count(logical=False),
        'logical_cores': psutil.cpu_count(logical=True),
        'max_threads': multiprocessing.cpu_count(),
        'current_process_cores': len(psutil.Process().cpu_affinity()) if hasattr(psutil.Process(), 'cpu_affinity') else None,
        'cpu_usage_percent': psutil.cpu_percent(interval=1),
        'available_memory_gb': round(psutil.virtual_memory().available / (1024**3), 2)
    }
    return cpu_info

def print_cpu_info():
    """
    Print formatted CPU information
    """
    info = get_cpu_info()
    print("\nCPU Information:")
    print(f"Physical CPU Cores: {info['physical_cores']}")
    print(f"Logical CPU Cores: {info['logical_cores']}")
    print(f"Maximum Available Threads: {info['max_threads']}")
    if info['current_process_cores'] is not None:
        print(f"Cores Available to Current Process: {info['current_process_cores']}")
    print(f"Current CPU Usage: {info['cpu_usage_percent']}%")
    print(f"Available Memory: {info['available_memory_gb']} GB")
    
    # Check if CUDA is available (for comparison)
    if torch.cuda.is_available():
        print("\nGPU Information:")
        print(f"CUDA Available: Yes")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("\nGPU Information:")
        print("CUDA Available: No")

def get_optimal_worker_count():
    """
    Calculate optimal number of workers for parallel processing.
    Returns:
        int: Recommended number of workers
    """
    # Get physical core count
    physical_cores = psutil.cpu_count(logical=False)
    
    # Leave one core free for system processes
    recommended_workers = max(1, physical_cores - 1)
    
    return recommended_workers
  
  
if __name__ == '__main__':
    print_cpu_info()
    print(get_optimal_worker_count())