"""
Utility Functions
Logging setup and image conversion helpers
"""

import base64
import io
import logging
import sys
from typing import Optional

import torch
from PIL import Image


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Setup colored logging for the application
    
    Args:
        level: Logging level (default: INFO)
    
    Returns:
        Logger instance
    """
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Return application logger
    logger = logging.getLogger(__name__)
    logger.info("📋 Logging initialized")
    
    return logger


def tensor_to_base64(tensor: torch.Tensor, format: str = "PNG") -> str:
    """
    Convert PyTorch tensor to base64-encoded image string
    
    Args:
        tensor: Image tensor with shape [C, H, W] in range [0, 1]
        format: Image format (PNG, JPEG, etc.)
    
    Returns:
        Base64-encoded string with data URI prefix
    """
    # Convert tensor to PIL Image
    # tensor is [C, H, W] in range [0, 1]
    tensor = tensor.clamp(0, 1)
    
    # Convert to numpy and transpose to [H, W, C]
    if tensor.dim() == 3:
        img_array = (tensor.cpu().numpy() * 255).astype('uint8')
        img_array = img_array.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
    else:
        raise ValueError(f"Expected 3D tensor [C, H, W], got shape {tensor.shape}")
    
    # Create PIL Image
    if img_array.shape[2] == 1:
        img = Image.fromarray(img_array.squeeze(), mode='L')
    elif img_array.shape[2] == 3:
        img = Image.fromarray(img_array, mode='RGB')
    else:
        raise ValueError(f"Unsupported number of channels: {img_array.shape[2]}")
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # Return with data URI prefix
    mime_type = f"image/{format.lower()}"
    return f"data:{mime_type};base64,{img_base64}"


def base64_to_tensor(base64_string: str) -> torch.Tensor:
    """
    Convert base64-encoded image to PyTorch tensor
    
    Args:
        base64_string: Base64 string (with or without data URI prefix)
    
    Returns:
        Tensor with shape [C, H, W] in range [0, 1]
    """
    # Remove data URI prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64
    img_bytes = base64.b64decode(base64_string)
    
    # Load image
    img = Image.open(io.BytesIO(img_bytes))
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to tensor
    import torchvision.transforms as transforms
    transform = transforms.ToTensor()
    tensor = transform(img)  # [C, H, W] in range [0, 1]
    
    return tensor


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def get_gpu_memory_info() -> Optional[dict]:
    """
    Get GPU memory information
    
    Returns:
        Dict with memory stats or None if CUDA unavailable
    """
    if not torch.cuda.is_available():
        return None
    
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    total = torch.cuda.get_device_properties(0).total_memory
    
    return {
        "allocated": format_bytes(allocated),
        "reserved": format_bytes(reserved),
        "total": format_bytes(total),
        "allocated_percent": (allocated / total) * 100,
        "reserved_percent": (reserved / total) * 100
    }


# Example usage
if __name__ == "__main__":
    # Test logging
    logger = setup_logging()
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    # Test tensor conversion
    test_tensor = torch.rand(3, 256, 256)
    base64_img = tensor_to_base64(test_tensor)
    print(f"Base64 length: {len(base64_img)}")
    print(f"Starts with: {base64_img[:50]}...")
    
    # Test GPU info
    if torch.cuda.is_available():
        gpu_info = get_gpu_memory_info()
        print(f"GPU Memory: {gpu_info}")