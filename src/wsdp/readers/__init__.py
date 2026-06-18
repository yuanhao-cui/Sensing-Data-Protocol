from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

from wsdp.structure import CSIData
from .bfee_reader import BfeeReader
from .xrf_reader import XrfReader
from .elder_reader import ElderReader
from .zte_reader import ZTEReader

# ^^^ import future reader above ^^^


_READER_REGISTRY = {
    'widar': BfeeReader,
    'gait': BfeeReader,
    'xrf55': XrfReader,
    'elderAL': ElderReader,
    "zte": ZTEReader,
}


def get_reader_class(dataset: str):
    """
    return the proper reader class according to dataset
    """
    reader_cls = _READER_REGISTRY.get(dataset)

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


def get_all_reader_metadata(dataset: str) -> dict:
    """
    Get metadata from the reader for a specific dataset.

    Args:
        dataset: Dataset name

    Returns:
        dict: Reader metadata
    """
    reader_cls = get_reader_class(dataset)
    reader = reader_cls()
    return reader.get_metadata()


def _validate_input_dir(file_path: str) -> Path:
    """Return a valid input directory path or raise the historical error."""
    input_path = Path(file_path)

    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"invalid file path: {input_path}")

    return input_path


def _collect_data_files(input_path: Path) -> List[Path]:
    """Collect candidate data files while excluding label/truth sidecars."""
    return [f for f in input_path.rglob("*") if f.is_file() and "truth" not in f.name]


def _extend_csi_data(target: List[CSIData], data) -> None:
    """Append a reader result that may be one CSIData object or a list of them."""
    if isinstance(data, list):
        target.extend(data)
    else:
        target.append(data)


def _process_file(reader, file_path):
    """
    process function for concurrent reading
    """
    try:
        # Sniff: skip files that don't match this reader's format
        if not reader.sniff(str(file_path)):
            return file_path.name, None, "format_mismatch"

        data = reader.read_file(str(file_path))

        return file_path.name, data, None
    except Exception as e:
        return file_path.name, None, str(e)


def load_data(file_path: str, dataset: str) -> List[CSIData]:
    input_path = _validate_input_dir(file_path)
    files = _collect_data_files(input_path)

    if not files:
        raise IOError(f"no file in folder: {input_path}")

    reader_class = get_reader_class(dataset)
    reader = reader_class()
    csi_data_list = []
    skipped = 0

    # Fan out independent file reads while preserving per-file error reporting.
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(_process_file, reader, file_path): file_path for file_path in files}

        for future in as_completed(futures):
            file_name, data, err = future.result()

            if err is None:
                _extend_csi_data(csi_data_list, data)
                print(f"√ processed: {file_name}\n")
            elif err == "format_mismatch":
                skipped += 1
            else:
                print(f"× unable to process {file_name}: {err}\n")

    if skipped > 0:
        print(f"[Info] skipped {skipped} file(s) (format mismatch for {dataset} reader)")

    return csi_data_list
