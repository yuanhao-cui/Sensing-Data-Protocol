"""
Inference interface for CSI classification models.

Provides a simple predict() function for running inference
on preprocessed CSI data using trained models.
"""
import torch
import numpy as np

from typing import Optional
from .datasets import CSIDataset
from .models import CSIModel
from .utils import load_custom_model
from .utils.resize import resize_csi_to_fixed_length


def predict(
    data: np.ndarray,
    model_path: str,
    num_classes: int,
    custom_model_path: Optional[str] = None,
    device: Optional[str] = None,
    padding_length: Optional[int] = None,
    dataset_name: str = "",
    pipeline_steps: Optional[dict] = None,
) -> np.ndarray:
    """
    Run inference on CSI data using a trained model.

    Args:
        data: CSI data array of shape (N, T, F, A) or (T, F, A) for single sample.
              If 3D, it will be treated as a single sample.
        model_path: Path to the saved checkpoint (.pth file)
        num_classes: Number of output classes
        custom_model_path: Path to custom model Python file (optional)
        device: 'cuda' or 'cpu'. Auto-detected if None.
        padding_length: Target time length for padding/truncation.
            If None, reads from checkpoint metadata; falls back to 1500.
        dataset_name: Dataset policy name. ``widar`` and ``gait`` use
            amplitude-phase inputs automatically; other datasets use amplitude.
        pipeline_steps: Preprocessing pipeline used for training. Required to
            recognize prepared Widar/Gait z-score amplitude-phase channels.

    Returns:
        np.ndarray: Predicted class indices, shape (N,) or scalar for single sample

    Example:
        >>> csi = np.random.randn(100, 30, 3) + 1j * np.random.randn(100, 30, 3)
        >>> preds = predict(csi, "best_checkpoint_42.pth", num_classes=6)
        >>> print(preds)  # e.g., array([3])
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure 4D: (N, T, F, A)
    if data.ndim == 3:
        data = data[np.newaxis, ...]
    elif data.ndim != 4:
        raise ValueError(f"Expected 3D (T,F,A) or 4D (N,T,F,A) input, got shape {data.shape}")

    # Load checkpoint to retrieve padding_length if stored
    device_obj = torch.device(device)
    checkpoint = torch.load(model_path, map_location=device_obj)

    # Resolve padding_length: checkpoint > caller > fallback
    if padding_length is None:
        padding_length = checkpoint.get('padding_length', 1500)

    # Pad/truncate to target length
    samples = [data[i] for i in range(len(data))]
    samples = resize_csi_to_fixed_length(samples, target_length=padding_length)
    data = np.stack(samples, axis=0)

    # Apply the same dataset-driven representation used during training.
    inference_dataset = CSIDataset(
        data,
        np.zeros(len(data), dtype=np.int64),
        dataset_name=dataset_name,
        pipeline_steps=pipeline_steps,
    )
    tensor_data = inference_dataset.data_list

    # Load model
    if custom_model_path:
        model = load_custom_model(custom_model_path, num_classes)
    else:
        model = CSIModel(num_classes=num_classes)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device_obj)
    model.eval()

    # Run inference
    tensor_data = tensor_data.to(device_obj)
    with torch.no_grad():
        outputs = model(tensor_data)
        _, predicted = torch.max(outputs.data, 1)

    return predicted.cpu().numpy()


def predict_single(
    csi_array: np.ndarray,
    model_path: str,
    num_classes: int,
    custom_model_path: Optional[str] = None,
    device: Optional[str] = None,
    padding_length: Optional[int] = None,
    dataset_name: str = "",
    pipeline_steps: Optional[dict] = None,
) -> int:
    """
    Convenience function for single-sample inference.

    Args:
        csi_array: Single CSI sample of shape (T, F, A)
        model_path: Path to saved checkpoint
        num_classes: Number of output classes
        custom_model_path: Path to custom model file (optional)
        device: 'cuda' or 'cpu' (optional)
        padding_length: Target time length
        dataset_name: Dataset policy name.
        pipeline_steps: Preprocessing pipeline used for training.

    Returns:
        int: Predicted class index
    """
    result = predict(
        csi_array[np.newaxis, ...],
        model_path=model_path,
        num_classes=num_classes,
        custom_model_path=custom_model_path,
        device=device,
        padding_length=padding_length,
        dataset_name=dataset_name,
        pipeline_steps=pipeline_steps,
    )
    return int(result[0])
