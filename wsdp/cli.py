import argparse

from .core import pipeline
from .download import download


def _run_pipeline(args):
    pipeline(input_path=args.input_path,
             output_folder=args.output_folder,
             dataset=args.dataset,
             model_path=args.model_path
             )


def _download_pipeline(args):
    download(args.dataset_name, args.dest)


def main_cli():
    parser = argparse.ArgumentParser(description="wsdp CLI")
    subparser = parser.add_subparsers(dest="command", required=True, help="available commands")

    parser_run = subparser.add_parser("run", help="run pipeline")
    parser_run.add_argument("input_path", required=True, type=str, help="input data path")
    parser_run.add_argument("output_folder", required=True, type=str, help="output path")
    parser_run.add_argument("dataset", required=True, type=str, help="dataset name")
    parser_run.add_argument("model_path", "-m", type=str, help="path of custom model")
    parser_run.set_defaults(func=_run_pipeline)

    parser_download = subparser.add_parser("download", help="download datasets")
    parser_download.add_argument("dataset_name", type=str, help="dataset name")
    parser_download.add_argument("dest", type=str, help="destination path for storing dataset")
    parser_download.set_defaults(func=_download_pipeline)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
