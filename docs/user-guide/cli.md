# CLI Usage

WSDP provides a command-line interface for common tasks.

## Commands

### `wsdp run`

Run the full training pipeline.

```bash
wsdp run INPUT_PATH OUTPUT_FOLDER DATASET [OPTIONS]

Options:
  -m, --model-path PATH      Path to a custom model .py file
                             (the file must expose `model = YourModelClass`)
  --model TEXT               Registered model name (default: CSIModel),
                             e.g. THAT, WiFlexFormer, ResNet1D
  --model-kwargs JSON        Extra model constructor arguments,
                             e.g. '{"dropout": 0.3}'
  --algorithm-config PATH    YAML/JSON algorithm pipeline config
  --algorithm-preset TEXT    Algorithm preset name,
                             e.g. high_quality, fast, robust
  --reader TEXT              Registered reader name used to load input files
                             (default: same as DATASET)
  --lr, --learning-rate FLOAT  Learning rate
  -e, --epochs INT           Number of epochs
  -b, --batch-size INT       Batch size
  -c, --config PATH          YAML hyperparameter override
                             (top-level key = dataset name;
                             NOT an algorithm pipeline config)

Examples:
  wsdp run ./data/elderAL ./output elderAL
  wsdp run ./data/widar ./output widar --lr 0.001 --epochs 50
  wsdp run ./data/widar ./output widar --model THAT
  wsdp run ./data/widar ./output widar -m custom_model.py
  wsdp run ./data/widar ./output widar --algorithm-config my_algorithms.yaml
  wsdp run ./data/widar ./output widar --algorithm-preset high_quality
```

### `wsdp download`

Download datasets.

```bash
wsdp download DATASET_NAME DEST [OPTIONS]

Options:
  -e, --email TEXT        Email for authentication
  -p, --password TEXT     Password for authentication
  -t, --token TEXT        JWT token
  --ext TEXT              Comma-separated extensions to download (e.g. '.csv,.mat')

Examples:
  wsdp download elderAL ./data --email user@example.com --password 'yourpass'
  wsdp download widar ./data
  wsdp download gait ./data
  wsdp download xrf55 ./data
  wsdp download zte ./data --email user@example.com --password 'yourpass'
```

> ⚠️ **zte dataset**: Requires applying for access on the SDP platform first.
> Account credentials alone are not sufficient — you must submit an access request
> at [sdp8.org](https://sdp8.org) for the zte dataset specifically.

> 📝 **gait dataset**: Data is in Intel IWL5300 binary (.dat) format.
> Use `--ext .csv,.mat` to skip binary files (but note: gait has only .dat files).

### `wsdp list`

List available datasets.

```bash
wsdp list [--verbose]
```

### `wsdp --version`

Show version information.

```bash
wsdp --version
```

See [API Reference](../API_REFERENCE.md) for full documentation.
