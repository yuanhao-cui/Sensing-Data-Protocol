import torch.nn as nn
import importlib.util


def load_custom_model(model_file_path, num_classes):
    try:
        spec = importlib.util.spec_from_file_location("custom_model", model_file_path)
        if spec is None:
            raise FileNotFoundError(f"cannot parse file: {model_file_path}")

        custom_model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_model_module)

        model_class = custom_model_module.model
        if not issubclass(model_class, nn.Module):
            raise TypeError(f"custom model is not a subclass of torch.nn.Module")

        model = model_class(num_classes)

        return model
    except FileNotFoundError as e:
        raise FileNotFoundError(f"file does not exist: {model_file_path}") from e
    except AttributeError as e:
        raise AttributeError(f"model class 'model' is not defined in file: {model_file_path}") from e
    except TypeError as e:
        if "missing" in str(e) or "unexpected" in str(e):
            raise TypeError(
                f"model initialization parameter error: {str(e)}. Please check the __init__ parameters of the custom "
                f"model.") from e
        raise TypeError(f"custom model type error: {str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"load custom model error: {str(e)}") from e