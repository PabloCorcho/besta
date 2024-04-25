import sys
import os
import argparse
import subprocess
import multiprocessing
import configparser
import numpy as np


def main(argv):
    """Pipeline for running parameterized and full version of HBSPS.

    This pipeline first runs the parameterized version of the HBSPS model to determine
    the ages and metallicities of the SSPs and the velocity dispersion sigma. Then, the
    pipeline runs the full version of the model (see Dries et al. 2016,2017). Within
    this pipeline it is assumed that there exists a file 'HBSPS.ini' that contains the
    relevant options for runnin the model.

    Args:
            spectrum: filename of input spectrum (commandline).

    """

    # Process commandline arguments
    parser = argparse.ArgumentParser(
        description="Process one spectrum using the full sampling pipeline."
    )
    parser.add_argument(
        "spectrum", type=str, default=False, help="file name of input spectrum."
    )
    args = parser.parse_args()

    # Configure HBSPS with options in HBSPS.ini
    options = configure(args)

    # Create cosmosis configuration file for parameterized version model
    iniFileX, outputFileX = createCosmoSISiniX(options)

    # Run parameterized version model
    if options["nCores"] == 1:
        command = "cosmosis " + iniFileX
    else:
        command = "mpirun -np " + str(options["nCores"]) + " cosmosis --mpi " + iniFileX
    runCommand(command)

    # Run postprocess command
    command = (
        "postprocess --no-plots --outdir "
        + options["outputDir"].rstrip("/")
        + " "
        + outputFileX
    )
    runCommand(command)

    # Get MAP values
    MAPX = getMAPX(options["outputDir"])

    # Process MAP-values to get age-metallicity indices best fitting templates and MAP-value of sigma
    sigma, ageIndices, FeHIndices = processMAPX(options, MAPX)

    # Create cosmosis configuration file for full version model
    iniFile, outputFile = createCosmoSISini(options, sigma, ageIndices, FeHIndices)

    # Run full version model
    if options["nCores"] == 1:
        command = "cosmosis " + iniFile
    else:
        command = "mpirun -np " + str(options["nCores"]) + " cosmosis --mpi " + iniFile
    runCommand(command)

    # Run postprocess command
    command = (
        "postprocess --no-plots --outdir "
        + options["outputDir"].rstrip("/")
        + " "
        + outputFile
    )
    runCommand(command)


def configure(args):
    """Get spectrum from commandline and options from 'HBSPS.ini'."""

    # Define dictionary with options
    options = dict()

    # Get input spectrum and check if it is valid
    if not os.path.isfile(args.spectrum):
        raise sys.exit(
            "\nERROR: File '"
            + args.spectrum
            + "' with input spectrum does not exist.\n"
        )
    inputData = np.loadtxt(args.spectrum)
    if inputData.shape[1] != 3:
        raise sys.exit(
            "\nERROR: File with input spectrum should contain three columns: (1) wavelength, (2) flux and (3) error spectrum.\n"
        )
    options["spectrum"] = args.spectrum

    # Read and check validity options in HBSPS.ini
    Config = configparser.ConfigParser()
    Config.read("HBSPS.ini")

    # Stellar templates
    options["templatesDir"] = Config.get("templates", "templatesDir")
    if not os.path.isdir(options["templatesDir"]):
        raise sys.exit(
            "\nERROR: Directory with stellar templates '"
            + options["templatesDir"]
            + "' is not valid.\n"
        )
    try:
        options["nSSPs"] = Config.getint("templates", "nSSPs")
        if options["nSSPs"] < 1:
            raise ValueError
    except ValueError:
        raise sys.exit(
            "\nERROR: nSSPs should be an integer larger than or equal to one.\n"
        )

    # IMF prior parameterization
    try:
        options["nSlopes"] = Config.getint("IMFprior", "nSlopes")
        if options["nSlopes"] != 1 and options["nSlopes"] != 2:
            raise ValueError
    except ValueError:
        raise sys.exit("\nERROR: nSlopes can only be 1 or 2.\n")
    try:
        options["regScheme"] = Config.getint("IMFprior", "regScheme")
        if options["regScheme"] != 1 and options["regScheme"] != 2:
            raise ValueError
    except ValueError:
        raise sys.exit(
            "\nERROR: regScheme can only be 1 (identity matrix) or 2 (1/w0**2).\n"
        )

    # Abundance variations
    try:
        options["sampleMg"] = Config.getboolean("responseFunctions", "sampleMg")
        options["sampleCa"] = Config.getboolean("responseFunctions", "sampleCa")
        options["sampleSi"] = Config.getboolean("responseFunctions", "sampleSi")
        options["sampleTi"] = Config.getboolean("responseFunctions", "sampleTi")
        options["sampleNa"] = Config.getboolean("responseFunctions", "sampleNa")
    except ValueError:
        raise sys.exit(
            "\nERROR: sampleMg, sampleCa, sampleSi, sampleTi and sampleNa should be boolean values.\n"
        )
    if (
        options["sampleMg"]
        or options["sampleCa"]
        or options["sampleSi"]
        or options["sampleTi"]
        or options["sampleNa"]
    ):
        options["resFunHDF5"] = Config.get("responseFunctions", "hdf5File")
        if not os.path.isfile(options["resFunHDF5"]):
            raise sys.exit(
                "\nERROR: Response functions file '"
                + options["resFunHDF5"]
                + "' does not exits.\n"
            )

    # Multiplicative polynomial
    try:
        options["polOrder"] = Config.getint("polynomial", "order")
        if options["polOrder"] < 0:
            raise ValueError
    except ValueError:
        raise sys.exit(
            "\nERROR: polynomial order should be an integer larger than or equal to zero.\n"
        )

    # Value of logbcov for parameterized version model
    try:
        options["logbcov"] = Config.getfloat("covariance", "logbcov")
    except ValueError:
        raise sys.exit("\nERROR: logbcov should be a float value.\n")

    # Output directory
    options["outputDir"] = Config.get("sampling", "outputDir")
    if not os.path.isdir(options["outputDir"]):
        raise sys.exit(
            "\nERROR: Output directory '" + options["outputDir"] + "' is not valid.\n"
        )

    # Details for sampling procedure
    try:
        options["nCores"] = Config.getint("sampling", "nCores")
        if options["nCores"] < 1:
            raise ValueError
    except ValueError:
        raise sys.exit(
            "\nERROR: nCores should be an integer larger than or equal to one.\n"
        )
    if options["nCores"] > multiprocessing.cpu_count():
        raise sys.exit(
            "\nERROR: nCores cannot be larger than available number of CPU's on this machine.\n"
        )
    try:
        options["livepoints"] = Config.getint("sampling", "livepoints")
        if options["livepoints"] < 1:
            raise ValueError
    except ValueError:
        raise sys.exit(
            "\nERROR: number of livepoints should be larger than or equal to one.\n"
        )

    # Priors sampled parameters
    try:
        if options["nSlopes"] == 1:
            options["alphaPr"] = np.asarray(
                Config.get("priors", "alpha").split(), dtype=float
            )
            if len(options["alphaPr"]) != 3:
                raise ValueError
        else:
            options["alpha1Pr"] = np.asarray(
                Config.get("priors", "alpha1").split(), dtype=float
            )
            options["alpha2Pr"] = np.asarray(
                Config.get("priors", "alpha2").split(), dtype=float
            )
            if len(options["alpha1Pr"]) != 3 or len(options["alpha2Pr"]) != 3:
                raise ValueError
    except ValueError:
        raise sys.exit("\nERROR: prior(s) IMF slopes not clear.\n")
    try:
        options["normPr"] = np.asarray(
            Config.get("priors", "norm").split(), dtype=float
        )
        if len(options["normPr"]) != 3:
            raise ValueError
    except ValueError:
        raise sys.exit("\nERROR: prior IMF normalization not clear.\n")
    try:
        options["sigmaPr"] = np.asarray(
            Config.get("priors", "sigma").split(), dtype=float
        )
        if len(options["sigmaPr"]) != 3:
            raise ValueError
    except ValueError:
        raise sys.exit("\nERROR: prior sigma not clear.\n")
    try:
        options["agePr"] = np.asarray(Config.get("priors", "age").split(), dtype=float)
        options["FeHPr"] = np.asarray(Config.get("priors", "FeH").split(), dtype=float)
        if len(options["agePr"]) != 3 or len(options["FeHPr"]) != 3:
            raise ValueError
    except ValueError:
        raise sys.exit("\nERROR: priors age/metallicity not clear.\n")
    try:
        options["logbcovPr"] = np.asarray(
            Config.get("priors", "logbcov").split(), dtype=float
        )
        if len(options["logbcovPr"]) != 3:
            raise ValueError
    except ValueError:
        raise sys.exit("\nERROR: prior logbcov not clear.\n")
    try:
        options["dexPr"] = np.asarray(Config.get("priors", "dex").split(), dtype=float)
        if len(options["dexPr"]) != 3:
            raise ValueError
    except ValueError:
        raise sys.exit("\nERROR: prior abundance variations not clear.\n")

    return options


def createCosmoSISiniX(options):
    """Create CosmoSIS ini-files for parameterized version model.

    Args:
            options: python dictionary with input options

    """
    # Remove extension and path from input spectrum
    fBasis = os.path.splitext(os.path.basename(options["spectrum"]))[0]

    # Determine file names of ini-file, output-file and values-file
    iniFile = os.path.join(
        options["outputDir"], fBasis + "-MX-" + str(options["nSSPs"]) + "SSPs.ini"
    )
    outputFile = os.path.join(
        options["outputDir"], fBasis + "-MX-" + str(options["nSSPs"]) + "SSPs.txt"
    )
    valuesFile = os.path.join(
        options["outputDir"],
        "values-" + fBasis + "-MX-" + str(options["nSSPs"]) + "SSPs.ini",
    )

    # Create ini-file for running Multinest
    with open(iniFile, "w") as fMX:
        fMX.write("[runtime]\n")
        fMX.write("sampler = multinest\n\n")
        fMX.write("[multinest]\n")
        fMX.write("max_iterations = 50000\n")
        fMX.write("live_points = " + str(options["livepoints"]) + "\n")
        fMX.write("feedback = True\n")
        fMX.write("tolerance = 1.0\n")
        fMX.write("update_interval = 200\n")
        fMX.write("log_zero = -1e14\n")
        fMX.write("multinest_outfile_root = " + options["outputDir"] + "\n\n")

        fMX.write("[output]\n")
        fMX.write("filename = " + outputFile + "\n")
        fMX.write("format = text\n\n")

        fMX.write("[pipeline]\n")
        fMX.write("modules = HBSPSPar\n")
        fMX.write("values = " + valuesFile + "\n")
        fMX.write("likelihoods = HBSPSPar\n")
        fMX.write("quiet = T\n")
        fMX.write("timing = F\n")
        fMX.write("debug = F\n\n")

        fMX.write("[HBSPSPar]\n")
        fMX.write("file = " + os.path.join(os.getcwd(), "HBSPSPar.py") + "\n")
        fMX.write("inputSpectrum = " + options["spectrum"] + "\n")
        fMX.write("templatesDir = " + options["templatesDir"] + "\n")
        fMX.write("nSSPs = " + str(options["nSSPs"]) + "\n")
        fMX.write("nSlopes = " + str(options["nSlopes"]) + "\n")
        fMX.write("logbcov = " + str(options["logbcov"]) + "\n")
        fMX.write("polOrder = " + str(options["polOrder"]) + "\n")
        fMX.write("sampleMg = " + str(options["sampleMg"]) + "\n")
        fMX.write("sampleCa = " + str(options["sampleCa"]) + "\n")
        fMX.write("sampleSi = " + str(options["sampleSi"]) + "\n")
        fMX.write("sampleTi = " + str(options["sampleTi"]) + "\n")
        fMX.write("sampleNa = " + str(options["sampleNa"]) + "\n")
        if (
            options["sampleMg"]
            or options["sampleCa"]
            or options["sampleSi"]
            or options["sampleTi"]
            or options["sampleNa"]
        ):
            fMX.write("resFunHDF5 = " + options["resFunHDF5"] + "\n")

    # Create values-file with sampled parameters and priors
    with open(valuesFile, "w") as fVX:
        fVX.write("[parameters]\n")
        if options["nSlopes"] == 1:
            fVX.write(
                "alpha = "
                + str(options["alphaPr"][0])
                + " "
                + str(options["alphaPr"][1])
                + " "
                + str(options["alphaPr"][2])
                + "\n"
            )
        else:
            fVX.write(
                "alpha1 = "
                + str(options["alpha1Pr"][0])
                + " "
                + str(options["alpha1Pr"][1])
                + " "
                + str(options["alpha1Pr"][2])
                + "\n"
            )
            fVX.write(
                "alpha2 = "
                + str(options["alpha2Pr"][0])
                + " "
                + str(options["alpha2Pr"][1])
                + " "
                + str(options["alpha2Pr"][2])
                + "\n"
            )
        fVX.write(
            "sigma = "
            + str(options["sigmaPr"][0])
            + " "
            + str(options["sigmaPr"][1])
            + " "
            + str(options["sigmaPr"][2])
            + "\n"
        )
        for tIdx in range(options["nSSPs"]):
            if tIdx >= 1:
                fVX.write("lumFr" + str(tIdx) + " = 0.001 0.5 1.0\n")
            fVX.write(
                "age"
                + str(tIdx + 1)
                + " = "
                + str(options["agePr"][0])
                + " "
                + str(options["agePr"][1])
                + " "
                + str(options["agePr"][2])
                + "\n"
            )
            fVX.write(
                "FeH"
                + str(tIdx + 1)
                + " = "
                + str(options["FeHPr"][0])
                + " "
                + str(options["FeHPr"][1])
                + " "
                + str(options["FeHPr"][2])
                + "\n"
            )
        if options["sampleMg"]:
            fVX.write(
                "MgDex = "
                + str(options["dexPr"][0])
                + " "
                + str(options["dexPr"][1])
                + " "
                + str(options["dexPr"][2])
                + "\n"
            )
        if options["sampleCa"]:
            fVX.write(
                "CaDex = "
                + str(options["dexPr"][0])
                + " "
                + str(options["dexPr"][1])
                + " "
                + str(options["dexPr"][2])
                + "\n"
            )
        if options["sampleSi"]:
            fVX.write(
                "SiDex = "
                + str(options["dexPr"][0])
                + " "
                + str(options["dexPr"][1])
                + " "
                + str(options["dexPr"][2])
                + "\n"
            )
        if options["sampleTi"]:
            fVX.write(
                "TiDex"
                + str(tIdx + 1)
                + " = "
                + str(options["dexPr"][0])
                + " "
                + str(options["dexPr"][1])
                + " "
                + str(options["dexPr"][2])
                + "\n"
            )
        if options["sampleNa"]:
            fVX.write(
                "NaDex = "
                + str(options["dexPr"][0])
                + " "
                + str(options["dexPr"][1])
                + " "
                + str(options["dexPr"][2])
                + "\n"
            )

    return iniFile, outputFile


def createCosmoSISini(options, sigma, ageIndices, FeHIndices):
    """Create CosmoSIS ini-files for full version model.

    Args:
            options: python dictionary with input options
            sigma: velocity dispersion from parameterized version model
            ageIndices: age-indices of stellar templates from parameterized version model
            FeHIndices: FeH-indices of stellar tempaltes from parameterized version model

    """
    # Remove extension and path from input spectrum
    fBasis = os.path.splitext(os.path.basename(options["spectrum"]))[0]

    # Determine file names of ini-file, output-file and values-file
    iniFile = os.path.join(
        options["outputDir"], fBasis + "-M-" + str(options["nSSPs"]) + "SSPs.ini"
    )
    outputFile = os.path.join(
        options["outputDir"], fBasis + "-M-" + str(options["nSSPs"]) + "SSPs.txt"
    )
    valuesFile = os.path.join(
        options["outputDir"],
        "values-" + fBasis + "-M-" + str(options["nSSPs"]) + "SSPs.ini",
    )

    # Create ini-file for running Multinest
    with open(iniFile, "w") as fM:
        fM.write("[runtime]\n")
        fM.write("sampler = multinest\n\n")
        fM.write("[multinest]\n")
        fM.write("max_iterations = 50000\n")
        fM.write("live_points = " + str(options["livepoints"]) + "\n")
        fM.write("feedback = True\n")
        fM.write("update_interval = 200\n")
        fM.write("log_zero = -1e14\n")
        fM.write("multinest_outfile_root = " + options["outputDir"] + "\n\n")

        fM.write("[output]\n")
        fM.write("filename = " + outputFile + "\n")
        fM.write("format = text\n\n")

        fM.write("[pipeline]\n")
        fM.write("modules = HBSPS\n")
        fM.write("values = " + valuesFile + "\n")
        fM.write("likelihoods = HBSPS\n")
        fM.write("quiet = T\n")
        fM.write("timing = F\n")
        fM.write("debug = F\n\n")

        fM.write("[HBSPS]\n")
        fM.write("file = " + os.path.join(os.getcwd(), "HBSPS.py") + "\n")
        fM.write("inputSpectrum = " + options["spectrum"] + "\n")
        fM.write("templatesDir = " + options["templatesDir"] + "\n")
        fM.write("nSSPs = " + str(options["nSSPs"]) + "\n")
        fM.write("nSlopes = " + str(options["nSlopes"]) + "\n")
        fM.write("polOrder = " + str(options["polOrder"]) + "\n")
        fM.write("sigma = " + str(sigma) + "\n")
        fM.write("ageIndices = ")
        for tIdx in range(options["nSSPs"]):
            fM.write(str(ageIndices[tIdx]) + " ")
        fM.write("\nFeHIndices = ")
        for tIdx in range(options["nSSPs"]):
            fM.write(str(FeHIndices[tIdx]) + " ")
        fM.write("\nsampleMg = " + str(options["sampleMg"]) + "\n")
        fM.write("sampleCa = " + str(options["sampleCa"]) + "\n")
        fM.write("sampleSi = " + str(options["sampleSi"]) + "\n")
        fM.write("sampleTi = " + str(options["sampleTi"]) + "\n")
        fM.write("sampleNa = " + str(options["sampleNa"]) + "\n")
        if (
            options["sampleMg"]
            or options["sampleCa"]
            or options["sampleSi"]
            or options["sampleTi"]
            or options["sampleNa"]
        ):
            fM.write("resFunHDF5 = " + options["resFunHDF5"] + "\n")

    # Create values-file with sampled parameters and priors
    with open(valuesFile, "w") as fV:
        fV.write("[parameters]\n")
        if options["nSlopes"] == 1:
            fV.write(
                "alpha = "
                + str(options["alphaPr"][0])
                + " "
                + str(options["alphaPr"][1])
                + " "
                + str(options["alphaPr"][2])
                + "\n"
            )
        else:
            fV.write(
                "alpha1 = "
                + str(options["alpha1Pr"][0])
                + " "
                + str(options["alpha1Pr"][1])
                + " "
                + str(options["alpha1Pr"][2])
                + "\n"
            )
            fV.write(
                "alpha2 = "
                + str(options["alpha2Pr"][0])
                + " "
                + str(options["alpha2Pr"][1])
                + " "
                + str(options["alpha2Pr"][2])
                + "\n"
            )
        for tIdx in range(options["nSSPs"]):
            fV.write(
                "norm"
                + str(tIdx + 1)
                + " = "
                + str(options["normPr"][0])
                + " "
                + str(options["normPr"][1])
                + " "
                + str(options["normPr"][2])
                + "\n"
            )
        fV.write(
            "logbcov = "
            + str(options["logbcovPr"][0])
            + " "
            + str(options["logbcovPr"][1])
            + " "
            + str(options["logbcovPr"][2])
            + "\n"
        )
        if options["sampleMg"]:
            fV.write(
                "MgDex = "
                + str(options["dexPr"][0])
                + " "
                + str(options["dexPr"][1])
                + " "
                + str(options["dexPr"][2])
                + "\n"
            )
        if options["sampleCa"]:
            fV.write(
                "CaDex = "
                + str(options["dexPr"][0])
                + " "
                + str(options["dexPr"][1])
                + " "
                + str(options["dexPr"][2])
                + "\n"
            )
        if options["sampleSi"]:
            fV.write(
                "SiDex = "
                + str(options["dexPr"][0])
                + " "
                + str(options["dexPr"][1])
                + " "
                + str(options["dexPr"][2])
                + "\n"
            )
        if options["sampleTi"]:
            fV.write(
                "TiDex"
                + str(tIdx + 1)
                + " = "
                + str(options["dexPr"][0])
                + " "
                + str(options["dexPr"][1])
                + " "
                + str(options["dexPr"][2])
                + "\n"
            )
        if options["sampleNa"]:
            fV.write(
                "NaDex = "
                + str(options["dexPr"][0])
                + " "
                + str(options["dexPr"][1])
                + " "
                + str(options["dexPr"][2])
                + "\n"
            )

    return iniFile, outputFile


def runCommand(command):
    """Run a command with subprocess"""
    subprocess.call(command, shell=True)


def getMAPX(outputDir):
    """Get MAP-values for parameterized version model."""
    fName = os.path.join(outputDir, "best_fit.txt")
    MAPX = np.loadtxt(fName, usecols=([1]))

    return MAPX


def processMAPX(options, MAPX):
    """Get age-metallicity indices and sigma from parameterized run of the model."""

    # Age-metallicity grid of templates (CHANGE IF YOU USE A DIFFERENT GRID)
    logAgeArr = np.linspace(9.175, 10.125, 39)
    FeHArr = np.linspace(-1.4, 0.4, 37)

    # Run over the different parameters
    parIdx = 1
    if options["nSlopes"] == 2:
        parIdx += 1
    sigma = MAPX[parIdx]
    parIdx += 1

    ageIndices = []
    FeHIndices = []
    lumFrs = []
    for tIdx in range(options["nSSPs"]):
        age = MAPX[parIdx]
        logAge = np.log10(age * 1e9)
        ageIndices.append(np.abs(logAgeArr - logAge).argmin())
        parIdx += 1
        FeH = MAPX[parIdx]
        FeHIndices.append(np.abs(FeHArr - FeH).argmin())
        parIdx += 1
        if tIdx < (options["nSSPs"] - 1):
            lumFrs.append(MAPX[parIdx])
            parIdx += 1
        else:
            lumFrs.append(1.0 - sum(lumFrs))

    # Order age-metallicity by value of lumFrs
    order = np.argsort(lumFrs)[::-1]
    ageIndices = np.asarray(ageIndices)
    FeHIndices = np.asarray(FeHIndices)
    ageIndices = ageIndices[order]
    FeHIndices = FeHIndices[order]

    return sigma, ageIndices, FeHIndices


if __name__ == "__main__":
    main(sys.argv[1:])
