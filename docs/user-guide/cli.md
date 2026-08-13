# CLI Usage

WSDP provides a command-line interface for common tasks.

## Commands

### `wsdp run`

Run the full training pipeline.

```bash
wsdp run INPUT_PATH OUTPUT_FOLDER DATASET [OPTIONS]

Options:
  -m, --model-path PATH    Path to custom model
  --lr FLOAT              Learning rate
  -e, --epochs INT        Number of epochs
  -b, --batch-size INT    Batch size
  -c, --config PATH       Config file path
  --reader TEXT           Registered reader name used to load input files
                          (default: same as DATASET)

Examples:
  wsdp run ./data/elderAL ./output elderAL
  wsdp run ./data/widar ./output widar --lr 0.001 --epochs 50
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
