import os
import sys
import argparse
from besta.io import Reader
from besta.postprocess import summarize_results

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
    return parser

def pprint(*mssgs):
    print("[BESTA]", *mssgs)

def load_reader(file, from_ini=False):
    if from_ini:
        pprint("Loading INI config file", file)
        reader = Reader(file)
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
    pprint("Plots generated successfully")

def make_corner_plot(file, from_ini=False):
    reader = load_reader(file, from_ini)
    pass

def make_chain_plots(file, from_ini=False):
    reader = load_reader(file, from_ini)
    pass

def make_summary_statistics(file, from_ini=False, output=None):
    reader = load_reader(file, from_ini)
    results = summarize_results(reader.results_table)
    results.write_fits(output, overwrite=True)

def interactive():
    # GUI to visalize the input data and (optionally) the models from the chains
    pass

def main():
    parser = parser_setup()
    args = parser.parse_args()
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

if __name__ == "__main__":
    main()
