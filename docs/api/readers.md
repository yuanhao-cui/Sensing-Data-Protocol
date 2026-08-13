# API Reference - Readers

See [Full API Reference](../API_REFERENCE.md) for complete documentation.

## Available Readers

| Reader | Dataset | Format |
|--------|---------|--------|
| BfeeReader | Widar / Gait | .dat (bfee) |
| XRF55Reader | XRF55 | .npy |
| ElderALReader | ElderAL | .csv |
| ZTEReader | ZTE | .csv |

## Usage

```python
from wsdp.readers import BfeeReader

reader = BfeeReader()
data = reader.read_file("/path/to/file.dat")
```

## Registering a Custom Reader

```python
from wsdp.readers import BaseReader, register_reader, create_reader

class MyReader(BaseReader):
    def sniff(self, file_path):
        return file_path.endswith(".myfmt")

    def read_file(self, file_path):
        ...  # parse the file into CSIData

register_reader("my_format", MyReader)  # replace=True to override a built-in
reader = create_reader("my_format")
```

Then pass `reader="my_format"` to `wsdp.pipeline()` to load files with it while
keeping a built-in dataset's filename convention for label/group parsing. See
`examples/scripts/custom_reader_algorithm.py` for a runnable example.
