import argparse
import os
import sys

from besta import io
from besta.pipeline import MainPipeline

def __print_alberti():
    text = """     -----------------------------------------------
     Nadie, nadie, nadie, que enfrente no hay nadie;
     que es nadie la muerte si va en tu montura.
     Galopa, caballo cuatralbo,
     jinete del pueblo,
     que la tierra es tuya. 
    \n        R. Alberti (1938, Galope)
    """
    print(text)

def _print(*parts):
    print("[BESTA-RUN]", *parts)


def _split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_list(value, *, name):
    raw_items = _split_csv(value)
    if not raw_items:
        raise ValueError(f"{name} cannot be empty")

    parsed = []
    for item in raw_items:
        try:
            parsed.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid integer in {name}: {item!r}") from exc
    return parsed


def _parse_optional_csv(value, *, name, expected_len):
    items = _split_csv(value)
    if len(items) != expected_len:
        raise ValueError(
            f"{name} must contain exactly {expected_len} entries; got {len(items)}"
        )
    return [os.path.expandvars(item) for item in items]


def _load_pipeline_config(path):
    path = os.path.expandvars(path)
    suffix = os.path.splitext(path)[1].lower()

    if suffix != ".ini":
        raise ValueError(
            "Unsupported configuration format. besta-run only accepts .ini files."
        )

    cfg = io._ini_file_to_dict(path)
    return [cfg]


def parser_setup():
    parser = argparse.ArgumentParser(
        description="Run BESTA fits through MainPipeline from a config file"
    )
    parser.add_argument(
        "config",
        help="Path to configuration file (.ini)",
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        default=1,
        help="Number of cores per subpipeline run (default: 1). Use -1 for all cores.",
    )
    parser.add_argument(
        "--n-cores-list",
        type=str,
        default=None,
        help="Comma-separated list of per-subpipeline cores (overrides --n-cores)",
    )
    parser.add_argument(
        "--ini-files",
        type=str,
        default=None,
        help="Comma-separated list of existing .ini files, one per subpipeline",
    )
    parser.add_argument(
        "--ini-values-files",
        type=str,
        default=None,
        help="Comma-separated list of values files, one per subpipeline",
    )
    parser.add_argument(
        "--plot-result",
        action="store_true",
        help="Generate best-fit plots for each module at the end of each run",
    )
    return parser


def run_fit(args):
    pipeline_config_list = _load_pipeline_config(args.config)
    n_subpipes = len(pipeline_config_list)

    if args.n_cores_list is not None:
        n_cores_list = _parse_int_list(args.n_cores_list, name="--n-cores-list")
        if len(n_cores_list) != n_subpipes:
            raise ValueError(
                f"--n-cores-list must have exactly {n_subpipes} entries; "
                f"got {len(n_cores_list)}"
            )
    else:
        n_cores_list = [int(args.n_cores)] * n_subpipes

    ini_files = None
    if args.ini_files is not None:
        ini_files = _parse_optional_csv(
            args.ini_files, name="--ini-files", expected_len=n_subpipes
        )
    elif n_subpipes == 1:
        # Reuse the user-provided ini directly for single-ini runs.
        ini_files = [os.path.expandvars(args.config)]

    ini_values_files = None
    if args.ini_values_files is not None:
        ini_values_files = _parse_optional_csv(
            args.ini_values_files,
            name="--ini-values-files",
            expected_len=n_subpipes,
        )

    _print(f"Loaded {n_subpipes} subpipeline configuration(s)")
    _print("Executing fit...")

    pipeline = MainPipeline(
        pipeline_config_list,
        n_cores_list=n_cores_list,
        ini_files=ini_files,
        ini_values_files=ini_values_files,
    )
    status = pipeline.execute_all(plot_result=args.plot_result)

    if int(status) == 0:
        _print("Pipeline execution completed successfully")
    else:
        _print(f"Pipeline execution failed with status code: {status}")

    return int(status)


def main():

    parser = parser_setup()
    args = parser.parse_args()

    try:
        status = run_fit(args)
        _print("Pipeline finished with status code:", status)
        __print_alberti()
    except Exception as exc:
        _print("Error:", str(exc))
        return 1
    return status


if __name__ == "__main__":
    sys.exit(main())
