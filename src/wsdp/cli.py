import argparse
import json
import os
import sys

from .core import pipeline
from .download import download, AUTH_REQUIRED_DATASETS, DownloadError
from .readers import list_datasets, get_all_reader_metadata
from . import __version__


def _run_pipeline(args: argparse.Namespace) -> None:
    # Build kwargs from CLI arguments
    kwargs = {}
    if args.learning_rate is not None:
        kwargs['learning_rate'] = args.learning_rate
    if args.epochs is not None:
        kwargs['num_epochs'] = args.epochs
    if args.batch_size is not None:
        kwargs['batch_size'] = args.batch_size
    if args.config is not None:
        kwargs['config_file'] = args.config
    if args.model is not None:
        kwargs['model_name'] = args.model
    if args.model_kwargs is not None:
        kwargs['model_kwargs'] = json.loads(args.model_kwargs)
    if args.algorithm_config is not None:
        kwargs['algorithm_config_file'] = args.algorithm_config
    if args.algorithm_preset is not None:
        kwargs['algorithm_preset'] = args.algorithm_preset
    if args.reader is not None:
        kwargs['reader'] = args.reader
    
    pipeline(
        input_path=args.input_path,
        output_folder=args.output_folder,
        dataset=args.dataset,
        model_path=args.model_path,
        **kwargs
    )


def _download_pipeline(args: argparse.Namespace) -> None:
    from .readers import list_datasets

    dataset = args.dataset_name
    datasets = list_datasets()

    if dataset not in datasets:
        print(f"error: unknown dataset '{dataset}'. Available: {', '.join(datasets)}")
        return

    if dataset in AUTH_REQUIRED_DATASETS and not args.token and not os.environ.get("WSDP_TOKEN"):
        if not args.email or not args.password:
            print(f"error: dataset '{dataset}' requires authentication.")
            print("Please provide --email and --password, or --token / WSDP_TOKEN env var.")
            print()
            print("Examples:")
            print(f"  wsdp download {dataset} ./data --email user@example.com --password 'yourpass'")
            print(f"  wsdp download {dataset} ./data --token <jwt_token>")
            print(f"  export WSDP_TOKEN=<jwt_token> && wsdp download {dataset} ./data")
            return

    # Parse extensions filter
    extensions = None
    if args.ext:
        extensions = [e.strip() for e in args.ext.split(',')]
        if not all(e.startswith('.') for e in extensions):
            print("error: --ext values must start with '.' (e.g. '.csv,.mat')")
            return

    try:
        download(
            args.dataset_name,
            args.dest,
            email=args.email,
            password=args.password,
            token=args.token,
            extensions=extensions,
        )
    except DownloadError as e:
        print(f"error: {e}")
        sys.exit(1)


def _list_datasets(args: argparse.Namespace) -> None:
    datasets = list_datasets()
    if args.verbose:
        print(f"Available datasets ({len(datasets)}):\n")
        for name in datasets:
            try:
                meta = get_all_reader_metadata(name)
                print(f"  {name}")
                print(f"    Format: {meta.get('format', 'N/A')}")
                print(f"    Description: {meta.get('description', 'N/A')}")
                print(f"    Reader: {meta.get('reader', 'N/A')}")
                if meta.get('subcarriers'):
                    print(f"    Subcarriers: {meta['subcarriers']}")
                if meta.get('complex') is not None:
                    print(f"    Complex: {meta['complex']}")
                print()
            except Exception:
                print(f"  {name}  (metadata unavailable)\n")
    else:
        print("Available datasets:")
        for name in datasets:
            print(f"  - {name}")


def main_cli() -> None:
    parser = argparse.ArgumentParser(
        description="wsdp - Wi-Fi Sensing Data Processing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"wsdp {__version__}",
        help="Show version information",
    )

    subparser = parser.add_subparsers(dest="command", help="available commands")

    # Run command
    parser_run = subparser.add_parser("run", help="run pipeline")
    parser_run.add_argument("input_path", type=str, help="input data path")
    parser_run.add_argument("output_folder", type=str, help="output path")
    parser_run.add_argument("dataset", type=str, help="dataset name")
    parser_run.add_argument("--model-path", "-m", type=str, help="path of custom model")
    parser_run.add_argument(
        "--model", type=str, default=None,
        help="registered model name, e.g. CSIModel, cnn1dmodel, THAT"
    )
    parser_run.add_argument(
        "--model-kwargs", type=str, default=None,
        help='JSON object with extra model constructor arguments, e.g. \'{"dropout": 0.3}\''
    )
    parser_run.add_argument(
        "--algorithm-config", type=str, default=None,
        help="path to YAML/JSON algorithm pipeline config"
    )
    parser_run.add_argument(
        "--algorithm-preset", type=str, default=None,
        help="algorithm preset name, e.g. high_quality, fast, robust"
    )
    parser_run.add_argument(
        "--reader", type=str, default=None,
        help="registered reader name or dataset alias to use for loading input files"
    )
    parser_run.add_argument(
        "--lr", "--learning-rate", dest="learning_rate", type=float, default=None,
        help="learning rate for training (default: from model_params.json)"
    )
    parser_run.add_argument(
        "--epochs", "-e", type=int, default=None,
        help="number of training epochs (default: from model_params.json)"
    )
    parser_run.add_argument(
        "--batch-size", "-b", type=int, default=None,
        help="batch size for training (default: from model_params.json)"
    )
    parser_run.add_argument(
        "--config", "-c", type=str, default=None,
        help="path to YAML config file for hyperparameter override"
    )
    parser_run.set_defaults(func=_run_pipeline)

    # Download command
    parser_download = subparser.add_parser("download", help="download datasets")
    parser_download.add_argument("dataset_name", type=str, help="dataset name")
    parser_download.add_argument("dest", type=str, help="destination path for storing dataset")
    parser_download.add_argument(
        "--email", "-e", type=str, default=None,
        help="Email for authentication (non-interactive mode)"
    )
    parser_download.add_argument(
        "--password", "-p", type=str, default=None,
        help="Password for authentication (non-interactive mode)"
    )
    parser_download.add_argument(
        "--token", "-t", type=str, default=None,
        help="JWT token for authentication (env var: WSDP_TOKEN)"
    )
    parser_download.add_argument(
        "--ext", type=str, default=None,
        help="Comma-separated file extensions to download (e.g. '.csv,.mat'). "
             "Files with other extensions will be skipped."
    )
    parser_download.set_defaults(func=_download_pipeline)

    # List command
    parser_list = subparser.add_parser("list", help="list available datasets")
    parser_list.add_argument(
        "--verbose", "-V", action="store_true",
        help="Show detailed metadata for each dataset"
    )
    parser_list.set_defaults(func=_list_datasets)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
