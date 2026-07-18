from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Type, Union

from wsdp.structure import CSIData
from .base import BaseReader
from .bfee_reader import BfeeReader
from .xrf_reader import XrfReader
from .elder_reader import ElderReader
from .zte_reader import ZTEReader

# ^^^ import future reader above ^^^


_READER_REGISTRY: dict[str, Type[BaseReader]] = {
    'widar': BfeeReader,
    'gait': BfeeReader,
    'xrf55': XrfReader,
    'elderAL': ElderReader,
    "zte": ZTEReader,
}

_READER_ALIASES: dict[str, str] = {}


def _canonical_dataset(dataset: str) -> str:
    """Resolve a dataset name or alias to its canonical registry key."""
    if dataset in _READER_REGISTRY:
        return dataset
    normalized = dataset.strip()
    if normalized in _READER_REGISTRY:
        return normalized
    return _READER_ALIASES.get(normalized.lower(), normalized)


def register_reader(
    dataset: str,
    reader_class: Type[BaseReader],
    *,
    aliases: list[str] | None = None,
    replace: bool = False,
) -> None:
    """Register a dataset reader so raw-data loading is pluggable.

    Args:
        dataset: Canonical dataset name.
        reader_class: ``BaseReader`` subclass that handles the format.
        aliases: Optional list of alternative names that map to this dataset.
        replace: If True, allow replacing an existing registration.

    Raises:
        TypeError: If ``reader_class`` is not a ``BaseReader`` subclass.
        ValueError: If the dataset is already registered and ``replace`` is False.
    """
    if not isinstance(reader_class, type) or not issubclass(reader_class, BaseReader):
        raise TypeError("reader_class must inherit from BaseReader")
    if dataset in _READER_REGISTRY and not replace:
        raise ValueError(f"reader already registered for dataset: {dataset}")
    _READER_REGISTRY[dataset] = reader_class
    for alias in aliases or []:
        _READER_ALIASES[alias.lower()] = dataset


def unregister_reader(dataset: str) -> bool:
    """Remove a reader registration, returning whether it existed."""
    canonical = _canonical_dataset(dataset)
    removed = _READER_REGISTRY.pop(canonical, None) is not None
    for alias, target in list(_READER_ALIASES.items()):
        if target == canonical:
            del _READER_ALIASES[alias]
    return removed


def create_reader(dataset: Union[str, Type[BaseReader], BaseReader]) -> BaseReader:
    """Create a reader instance for a dataset name, class, or instance."""
    if isinstance(dataset, type) and issubclass(dataset, BaseReader):
        return dataset()
    if isinstance(dataset, BaseReader):
        return dataset
    canonical = _canonical_dataset(str(dataset))
    reader_cls = _READER_REGISTRY.get(canonical)
    if reader_cls is None:
        raise ValueError(f"not supported dataset: {dataset}")
    return reader_cls()


def get_reader_class(dataset: str) -> Type[BaseReader]:
    """Return the proper reader class according to dataset."""
    canonical = _canonical_dataset(dataset)
    reader_cls = _READER_REGISTRY.get(canonical)
    if reader_cls is None:
        raise ValueError(f"not supported dataset: {dataset}")
    return reader_cls


def list_datasets() -> List[str]:
    """
    List all available dataset names.

    Returns:
        list: sorted list of dataset names
    """
    return sorted(_READER_REGISTRY.keys())


def list_readers() -> List[str]:
    """List all registered canonical dataset names."""
    return list_datasets()


def get_all_reader_metadata(dataset: str) -> dict:
    """
    Get metadata from the reader for a specific dataset.

    Args:
        dataset: Dataset name

    Returns:
        dict: Reader metadata
    """
    reader = create_reader(dataset)
    return reader.get_metadata()


def _process_file(reader: BaseReader, file_path: Path) -> tuple[str, list[CSIData], str | None]:
    """
    process function for concurrent reading
    """
    try:
        # Sniff: skip files that don't match this reader's format
        if not reader.sniff(str(file_path)):
            return file_path.name, [], "format_mismatch"
        data = reader.read_file(str(file_path))
        return file_path.name, data if isinstance(data, list) else [data], None
    except Exception as e:
        return file_path.name, [], str(e)


def load_data(file_path: str, dataset: str) -> List[CSIData]:
    input_path = Path(file_path)
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"invalid file path: {input_path}")
    files = [f for f in input_path.rglob("*") if f.is_file() and "truth" not in f.name]
    if not files:
        raise IOError(f"no file in folder: {input_path}")

    reader = create_reader(dataset)
    csi_data_list: list[CSIData] = []
    skipped = 0

    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(_process_file, reader, file_path): file_path for file_path in files}
        for future in as_completed(futures):
            file_name, data, err = future.result()
            if err is None:
                csi_data_list.extend(data)
                print(f"√ processed: {file_name}\n")
            elif err == "format_mismatch":
                skipped += 1
            else:
                print(f"× unable to process {file_name}: {err}\n")

    if skipped > 0:
        print(f"[Info] skipped {skipped} file(s) (format mismatch for {dataset} reader)")

    return csi_data_list


__all__ = [
    "BaseReader",
    "BfeeReader",
    "XrfReader",
    "ElderReader",
    "ZTEReader",
    "create_reader",
    "get_reader_class",
    "list_datasets",
    "list_readers",
    "get_all_reader_metadata",
    "load_data",
    "register_reader",
    "unregister_reader",
]
