import os
import sys
import argparse
from besta.io import Reader
from besta.postprocess import (
    summarize_results, to_physical_table,
    reconstruct_sfh_from_file, reconstruct_sfh_from_reader)

def parser_setup():
    parser = argparse.ArgumentParser(
        description="BESTA: Bayesian Estimator of Stellar Population Analysis"
    )
    parser.add_argument("file", help="Input file (results or ini)")
    parser.add_argument("--from_ini", action="store_true",
                        help="Indicate that the input file is an ini file")
    parser.add_argument("--make_best_fit", action="store_true",
                        help="Generate best fit plots for each module")
    parser.add_argument("--make_corner_plot", action="store_true",
                        help="Generate corner plot for the results")
    parser.add_argument("--make_summary_statistics", action="store_true",
                        help="Generate summary statistics from the results")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for summary statistics (FITS format)")
    sfh = parser.add_argument_group(
        "SFH reconstruction",
        "Posterior SFHs on a grid of lookback times, saved as FITS. Latent SFH "
        "parameters are converted to physical space automatically.")
    sfh.add_argument("--make_sfh", action="store_true",
                     help="Reconstruct the SFH and write it to a FITS file")
    sfh.add_argument("--sfh_output", type=str, default=None,
                     help="Output FITS file (default: <results>_sfh.fits)")
    sfh.add_argument("--sfh_plot", "--sfh-plot", action="store_true",
                     help="Generate SFH percentile plots")
    sfh.add_argument("--sfh_plot_output", "--sfh-plot-output", type=str, default=None,
                     help="Output plot file (default: <results>_sfh.png)")
    sfh.add_argument("--sfh_samples", action="store_true",
                     help="Also store the per-sample SFHs (not only percentiles)")
    sfh.add_argument("--sfh_n_bins", type=int, default=40,
                     help="Number of log-spaced lookback-time bins (default: 40)")
    sfh.add_argument("--sfh_min_lookback", type=float, default=1e-3,
                     help="Upper edge of the first lookback bin in Gyr (default: 1e-3)")
    sfh.add_argument("--sfh_lookback_edges", type=str, default=None,
                     help="Comma-separated lookback bin edges in Gyr, starting at 0 "
                          "(overrides --sfh_n_bins/--sfh_min_lookback)")
    sfh.add_argument("--sfh_taus", type=str, default="0.01,0.1,1.0",
                     help="Comma-separated timescales in Gyr for the average sSFR "
                          "(default: 0.01,0.1,1.0)")
    sfh.add_argument("--sfh_percentiles", type=str, default="0.05,0.16,0.5,0.84,0.95",
                     help="Comma-separated quantiles in [0, 1]")
    sfh.add_argument("--sfh_max_samples", type=int, default=300,
                     help="Use a random subset of at most this many samples")
    sfh.add_argument("--sfh_module", type=str, default=None,
                     help="Pipeline module defining the SFH (default: the first one)")
    sfh.add_argument("--burn_in", type=int, default=0,
                     help="Samples to discard per walker (default: 0)")
    sfh.add_argument("--nwalkers", type=int, default=1,
                     help="Number of walkers, used with --burn_in (default: 1)")
    return parser


def _floats(text):
    return [float(item) for item in text.split(",") if item.strip()]

def pprint(*mssgs):
    print("[BESTA]", *mssgs)

def load_reader(file, from_ini=False):
    if from_ini:
        pprint("Loading INI config file", file)
        reader = Reader(ini_file=file)
    else:
        pprint("Loading results file", file)
        reader= Reader.from_results_file(file)
    reader.load_results()
    return reader

def make_best_fit(file, from_ini=False):
    reader = load_reader(file, from_ini)
    pprint("Loading maximum a posteriori (MAP) solution")
    solution = reader.get_maxlike_solution()
    solution_datablock = reader.solution_to_datablock(solution)
    for par_module in reader.modules:
        pprint("Generating plot for module", par_module)
        pipeline_module = reader.get_module(par_module)
        figname = reader.ini["output"].get(
                    "figurename",
                    reader.ini["output"]["filename"].replace(".txt", "")
                    + f"_{par_module}_best_fit_solution.png",
                    )
        pipeline_module.plot_solution(solution_datablock,
                                      figname=figname)
        pprint("  Plot generated successfully at", figname)

def make_corner_plot(file, from_ini=False):
    reader = load_reader(file, from_ini)
    pass

def make_chain_plots(file, from_ini=False):
    reader = load_reader(file, from_ini)
    pass

def make_summary_statistics(file, from_ini=False, output=None):
    reader = load_reader(file, from_ini)
    # Latent SFH parameters (use_transforms = T) are always summarised in
    # physical space.
    table = to_physical_table(reader)
    results = summarize_results(table)
    results.write_fits(output, overwrite=True)

def make_sfh(file, args):
    """Reconstruct the posterior SFHs of a run and write them to FITS."""
    kwargs = dict(
        module_name=args.sfh_module,
        lookback_edges=(_floats(args.sfh_lookback_edges)
                        if args.sfh_lookback_edges else None),
        n_bins=args.sfh_n_bins,
        min_lookback=args.sfh_min_lookback,
        taus=_floats(args.sfh_taus),
        percentiles=_floats(args.sfh_percentiles),
        max_samples=args.sfh_max_samples,
        burn_in=args.burn_in,
        nwalkers=args.nwalkers,
    )
    if args.from_ini:
        reader = load_reader(file, from_ini=True)
        results = reader.ini["output"]["filename"]
        reconstruction = reconstruct_sfh_from_reader(reader, **kwargs)
    else:
        pprint("Loading results file", file)
        results = file
        reconstruction = reconstruct_sfh_from_file(file, **kwargs)
    
    if args.sfh_plot:
        plot_output = args.sfh_plot_output or os.path.splitext(results)[0] + "_sfh.png"
        from matplotlib import pyplot as plt
        fig, _ = reconstruction.make_figure()
        fig.savefig(plot_output, bbox_inches="tight")
        plt.close(fig)
        pprint(f"SFH percentile plot written to", plot_output)
    output = args.sfh_output or os.path.splitext(results)[0] + "_sfh.fits"
    reconstruction.write_fits(output, include_samples=args.sfh_samples)
    pprint(f"SFH of {reconstruction.n_samples} samples written to", output)
    return output

def interactive():
    # GUI to visalize the input data and (optionally) the models from the chains
    pass

def main(argv=None):
    parser = parser_setup()
    args = parser.parse_args(argv)
    file = args.file
    from_ini = args.from_ini
    if args.make_best_fit:
        print("Creating best-fit model plot")
        make_best_fit(file, from_ini)
    if args.make_corner_plot:
        print("Creating corner plot")
        make_corner_plot(file, from_ini)
    if args.make_summary_statistics:
        print("Creating summary statistics FITS")
        output = args.output
        make_summary_statistics(file, from_ini, output)
    if args.make_sfh:
        print("Reconstructing the star formation history")
        make_sfh(file, args)

if __name__ == "__main__":
    main()
