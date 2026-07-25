"""WSDP Model Library - Pluggable model registry and unified API."""

from .registry import (
    MODEL_REGISTRY,
    register_model,
    unregister_model,
    get_model,
    list_models,
    create_model,
)
from .csi_model import CSIModel
from .baselines import MLPModel, CNN1DModel, CNN2DModel, LSTMModel
from .mainstream import ResNet1D, ResNet2D, BiLSTMAttention, EfficientNetCSI
from .sota import VisionTransformerCSI, MambaCSI, GraphNeuralCSI
from .specialized import THAT, CSITime, PA_CSI
from .lightweight import WiFlexFormer, AttentionGRU
from .cross_domain import EI, FewSense

__all__ = [
    # Registry
    "MODEL_REGISTRY",
    "register_model",
    "unregister_model",
    "get_model",
    "list_models",
    "create_model",
    # Baseline models
    "MLPModel", "CNN1DModel", "CNN2DModel", "LSTMModel",
    # Mainstream models
    "ResNet1D", "ResNet2D", "BiLSTMAttention", "EfficientNetCSI",
    # SOTA models
    "VisionTransformerCSI", "MambaCSI", "GraphNeuralCSI",
    # Specialized CSI models
    "THAT", "CSITime", "PA_CSI",
    # Lightweight models
    "WiFlexFormer", "AttentionGRU",
    # Cross-domain models
    "EI", "FewSense",
    # Original model (backward compatible)
    "CSIModel",
]
