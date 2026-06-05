__version__ = "0.4.0"

__all__ = ["pipeline", "download", "predict", "predict_single"]


def __getattr__(name):
    """Load heavyweight public helpers only when callers ask for them."""
    if name == "pipeline":
        from .core import pipeline

        globals()[name] = pipeline
        return pipeline
    if name == "download":
        from .download import download

        globals()[name] = download
        return download
    if name in ("predict", "predict_single"):
        from .inference import predict, predict_single

        globals()["predict"] = predict
        globals()["predict_single"] = predict_single
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
