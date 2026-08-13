# API Reference - Core

## `wsdp.pipeline()`

Main training pipeline. Handles data loading, preprocessing, model training, and evaluation.

```python
from wsdp import pipeline

pipeline(
    input_path,
    output_folder,
    dataset,
    model_path=None,
    model_name="CSIModel",
    model_kwargs=None,
    pipeline_steps=None,
    algorithm_config_file=None,
    algorithm_preset=None,
    reader=None,
    batch_size=None,
    learning_rate=None,
    weight_decay=None,
    num_epochs=None,
    padding_length=None,
    test_split=0.3,
    val_split=0.5,
    num_seeds=5,
    config_file=None,
    num_workers=None,
    progress_callback=None,
    use_cache=True,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` | (required) | Path to dataset directory |
| `output_folder` | `str` | (required) | Path for outputs (model checkpoints, logs, plots) |
| `dataset` | `str` | (required) | Dataset name (`elderAL`, `widar`, `gait`, `xrf55`, `zte`) |
| `model_path` | `str` | `None` | Path to custom model file (`.py` exposing `model = YourModelClass`; takes precedence over `model_name`) |
| `model_name` | `str` | `"CSIModel"` | Registered model name |
| `model_kwargs` | `dict` | `None` | Extra constructor arguments forwarded to the model |
| `pipeline_steps` | `dict` | `None` | Flat algorithm config, e.g. `{'denoise': {'method': 'wavelet', 'level': 2}}` |
| `algorithm_config_file` | `str` | `None` | Path to YAML/JSON algorithm pipeline config |
| `algorithm_preset` | `str` | `None` | Algorithm preset name (`high_quality`, `fast`, `robust`, ...) |
| `reader` | `str` | `None` | Registered reader name (defaults to the dataset name) |
| `batch_size` | `int` | `None` | Training batch size (overrides config) |
| `learning_rate` | `float` | `None` | Optimizer learning rate (overrides config) |
| `weight_decay` | `float` | `None` | Weight decay (overrides config) |
| `num_epochs` | `int` | `None` | Number of training epochs (overrides config) |
| `padding_length` | `int` | `None` | Fixed sequence length for padding/truncation |
| `test_split` | `float` | `0.3` | Fraction of data for test set |
| `val_split` | `float` | `0.5` | Fraction of hold-out data for validation set |
| `num_seeds` | `int` | `5` | Number of random seeds to average over |
| `config_file` | `str` | `None` | Path to YAML hyperparameter config file |
| `num_workers` | `int` | `None` | Number of data loading workers for `DataLoader` (auto-detects if None) |
| `progress_callback` | `callable` | `None` | Called with `(epoch, total_epochs, metrics_dict)` after each epoch. |
| `use_cache` | `bool` | `True` | Cache preprocessed data to disk. Speeds up repeated runs on the same dataset. |

Algorithm pipeline resolution order: `pipeline_steps` > `algorithm_config_file` >
`algorithm_preset` > default chain (linear calibration → wavelet denoise).

**Returns:** `None` (outputs are saved to `output_folder`)

## `wsdp.predict()`

Run inference on CSI data using a trained model checkpoint.

```python
from wsdp import predict

predictions = predict(
    data,
    model_path,
    num_classes,
    custom_model_path=None,
    device=None,
    padding_length=None,
    dataset_name="",
    pipeline_steps=None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `np.ndarray` | (required) | CSI tensor of shape `(N, T, F, A)` or `(T, F, A)` for a single sample, real or complex |
| `model_path` | `str` | (required) | Path to saved model checkpoint (`.pth`) |
| `num_classes` | `int` | (required) | Number of output classes |
| `custom_model_path` | `str` | `None` | Path to the custom model `.py` file, if the checkpoint was trained with one |
| `device` | `str` | `None` | `'cuda'` or `'cpu'` (auto-detected when omitted) |
| `padding_length` | `int` | `None` | Pads/truncates the time dimension; reads from checkpoint metadata when omitted (fallback: 1500) |
| `dataset_name` | `str` | `""` | Dataset policy name; `widar`/`gait` use amplitude-phase inputs automatically |
| `pipeline_steps` | `dict` | `None` | The preprocessing config used at training time |

**Returns:** `np.ndarray` of predicted class indices, shape `(N,)`

## `wsdp.download()`

Download datasets from [SDP8.org](https://sdp8.org).

```python
from wsdp import download

download(dataset_name, dest, email=None, password=None, token=None, extensions=None)
```

## Internal Helpers

These functions are used internally by `pipeline()` but can be called directly for custom workflows.

### `_load_and_preprocess()`

```python
from wsdp.core import _load_and_preprocess

processed_data, labels, groups, unique_labels = _load_and_preprocess(
    input_path,
    dataset,
    pad_len=1500,          # required positional argument
    pipeline_steps=None,   # optional flat algorithm config
    reader=None,           # optional registered reader name
)
```

Loads raw CSI files, applies the algorithm pipeline (denoising, calibration, normalization, interpolation), and returns preprocessed tensors.

### `_create_data_split()`

```python
from wsdp.core import _create_data_split

train_data, val_data, test_data, train_labels, val_labels, test_labels = _create_data_split(
    processed_data, labels, groups,
    test_split=0.3,        # required
    val_split=0.5,         # required
    seed=42,               # required
    use_simple_split=False # required
)
```

Splits data into train/val/test sets. Returns numpy arrays ready for `DataLoader` wrapping.

### `_evaluate_model()`

```python
from wsdp.core import _evaluate_model

predictions, labels, accuracy = _evaluate_model(model, test_loader, device)
```

Evaluates a trained model on a test set. Returns a tuple of `(predictions, labels, accuracy)`.

See the [Full API Reference](../API_REFERENCE.md) for complete documentation.
