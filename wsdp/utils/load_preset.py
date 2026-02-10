import json

from importlib.resources import files
from wsdp import configs


def load_params(dataset_name: str) -> dict:
    json_path = files(configs) / "model_params.json"

    json_content = json_path.read_text(encoding="utf-8")
    data = json.loads(json_content)

    if dataset_name not in data:
        raise ValueError(f"no such dataset_name in config file: {dataset_name}. \
                            all dataset available: {list(data.keys())}")

    return data[dataset_name]


def load_api(key: str) -> str:
    json_path = files(configs) / "api.json"

    json_content = json_path.read_text(encoding="utf-8")
    api = json.loads(json_content)
    return api[key]


def load_mapping(dataset_name: str) -> str:
    json_path = files(configs) / "mapping.json"

    json_content = json_path.read_text(encoding="utf-8")
    mapping = json.loads(json_content)

    if dataset_name not in mapping:
        raise ValueError(f"no such dataset in config file: {dataset_name}. \
                            all dataset available: {list(mapping.keys())}")

    return mapping[dataset_name]
