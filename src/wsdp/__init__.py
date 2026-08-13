from .core import pipeline
from .download import download
from .inference import predict, predict_single

__version__ = "0.5.2"

__all__ = ["pipeline", "download", "predict", "predict_single"]
