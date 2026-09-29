"""Convert BESTA results tables from latent to physical SFH space.

Example
-------
    besta-to-physical results.txt                  # -> results_physical.txt
    besta-to-physical results.txt -o physical.txt
"""
import argparse
import sys

from besta.logging import setup_logging


def _print(*parts):
    print("[BESTA-TO-PHYSICAL]", *parts)


def parser_setup():
    parser = argparse.ArgumentParser(
        description=(
            "Write a copy of a CosmoSIS text results file with the latent SFH "
            "parameters (use_transforms = T) converted to physical values."
        )
    )
    parser.add_argument("results", nargs="+",
                        help="CosmoSIS text results file(s)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file (only with a single input). "
                             "Default: <results>_physical.txt")
    parser.add_argument("--module", default=None,
                        help="Only convert the SFH of this pipeline module")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="Fail if the output file already exists")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print errors")
    return parser


def main(argv=None):
    args = parser_setup().parse_args(argv)
    if args.output is not None and len(args.results) > 1:
        _print("Error: --output can only be used with a single input file")
        return 2

    setup_logging(level="WARNING" if args.quiet else "INFO", console=True)
    from besta.postprocess import convert_results_file

    status = 0
    for path in args.results:
        try:
            output = convert_results_file(
                path, args.output, module_name=args.module,
                overwrite=not args.no_overwrite)
        except Exception as exc:  # report and continue with the next file
            _print(f"Error converting {path}: {exc}")
            status = 1
            continue
        if output is None:
            _print(f"{path}: nothing to convert (already physical or no latent SFH)")
        else:
            _print(f"{path} -> {output}")
    return status


if __name__ == "__main__":
    sys.exit(main())
