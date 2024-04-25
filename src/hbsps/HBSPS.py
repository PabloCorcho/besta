import sys
import numpy as np
import copy
import cosmosis
from cosmosis.datablock import option_section, names as section_names
import hbsps.specBasics as specBasics
import modules.SPSbasics as SPSbasics
import modules.linearInv as linearInv


def setup(options):
    """Set-up the COSMOSIS sampler.

    Args:
            options: options from startup file (i.e. .ini file)
    Returns:
            config: parameters or objects that are passed to
                    the sampler.

    """

    # Read paramaters/options from start-up file
    fileName = options[option_section, "inputSpectrum"]
    templatesDir = options[option_section, "templatesDir"]
    nSSPs = options[option_section, "nSSPs"]
    nSlopes = options[option_section, "nSlopes"]
    ageIndices = options[option_section, "ageIndices"]
    FeHIndices = options[option_section, "FeHIndices"]
    sigma = options[option_section, "sigma"]
    polOrder = options[option_section, "polOrder"]
    abVariations = dict()
    abVariations["Mg"] = options[option_section, "sampleMg"]
    abVariations["Ca"] = options[option_section, "sampleCa"]
    abVariations["Si"] = options[option_section, "sampleSi"]
    abVariations["Ti"] = options[option_section, "sampleTi"]
    abVariations["Na"] = options[option_section, "sampleNa"]
    if any(value == True for value in list(abVariations.values())):
        resFunHDF5 = options[option_section, "resFunHDF5"]
    else:
        resFunHDF5 = None

    # Convert ageIndices / FeHIndices to numpy array if nSSPs == 1
    if nSSPs == 1:
        ageIndices = np.array([ageIndices])
        FeHIndices = np.array([FeHIndices])

    # Convert velocity dispersion to take into account that stellar
    # stellar templates are smoothed to 100.0 km/s.
    sigma = np.sqrt(sigma**2 - 100.0**2)

    # Read wavelength, SSP spectrum and create covariance matrix
    wavelength, flux, error = np.loadtxt(fileName, unpack=True)
    MF = 1.0 / np.average(flux)
    flux *= MF
    error *= MF
    cov = error * error

    # Select wavelength range that you want to use
    goodIdx = np.where(wavelength <= 8800)[0]
    wavelength = wavelength[goodIdx]
    flux = flux[goodIdx]
    cov = cov[goodIdx]
    error = error[goodIdx]
    nBins = len(wavelength)

    # Set available ages and metallicities.
    logAgeArr = np.linspace(9.175, 10.125, 39)
    FeHArr = np.linspace(-1.4, 0.4, 37)
    nAges = len(logAgeArr)
    nFeHs = len(FeHArr)

    # Create stTemplates-object for quickly reading SSP spectra.
    stTemplates = SPSbasics.stellarTemplates(templatesDir, logAgeArr, FeHArr)

    # Load response functions.
    respFunc = SPSbasics.respFunctions(
        nSSPs, abVariations, wavelength, goodIdx, resFunHDF5
    )

    # Create lists with numpy arrays of mass boundaries, templates
    # spectra and luminosities of all SSPs.
    massLowList = []
    massUpList = []
    templatesList = []
    for tIdx in range(nSSPs):
        massLow, massUp, spectra, lum = stTemplates.getTemplatesIdx(
            ageIndices[tIdx], FeHIndices[tIdx]
        )
        spectra = spectra[goodIdx, :]

        massLowList.append(massLow)
        massUpList.append(massUp)
        templatesList.append(spectra)

    # Create Salpeter IMF prior to initialize linearInv-object
    priorList = []
    for tIdx in range(nSSPs):
        prior = SPSbasics.powerlawIMFprior(
            massLowList[tIdx], massUpList[tIdx], alpha1=2.35, alpha2=2.35, norm=1.0
        )
        priorList.append(prior.weights)

    # Concatenate SSPs into CSP
    S = templatesList[0]
    massLow = massLowList[0]
    massUp = massUpList[0]
    w0 = priorList[0]
    for tIdx in range(nSSPs - 1):
        massLow = np.concatenate((massLow, massLowList[tIdx + 1]))
        massUp = np.concatenate((massUp, massUpList[tIdx + 1]))
        S = np.concatenate((S, templatesList[tIdx + 1]), 1)
        w0 = np.concatenate((w0, priorList[tIdx + 1]))

    # Initialize linearInversion.SSPobject
    linSol = linearInv.invertCSP(wavelength, flux, cov, S, w0, regScheme=1)

    # Basis of Legendre polynomials for multiplicative polynomial
    AL = specBasics.getLegendrePolynomial(wavelength, polOrder, bounds=None)

    # Pass parameters to execute function.
    config = {}
    config["flux"] = flux
    config["wavelength"] = wavelength
    config["nBins"] = nBins
    config["error"] = error
    config["cov"] = cov
    config["nSSPs"] = nSSPs
    config["nSlopes"] = nSlopes
    config["polOrder"] = polOrder
    config["AL"] = AL
    config["sigma"] = sigma
    config["ageIndices"] = ageIndices
    config["FeHIndices"] = FeHIndices
    config["massLowList"] = massLowList
    config["massUpList"] = massUpList
    config["templatesList"] = templatesList
    config["respFunc"] = respFunc
    config["linSol"] = linSol

    return config


def execute(block, config):
    """Function executed by sampler

    This is the function that is executed many times by the sampler. The
    likelihood resulting from this function is the evidence on the basis
    of which the parameter space is sampled.

    """

    # Obtain parameters from setup
    flux = config["flux"]
    wavelength = config["wavelength"]
    nBins = config["nBins"]
    error = config["error"]
    cov = config["cov"]
    nSSPs = config["nSSPs"]
    nSlopes = config["nSlopes"]
    polOrder = config["polOrder"]
    AL = config["AL"]
    sigma = config["sigma"]
    ageIndices = config["ageIndices"]
    FeHIndices = config["FeHIndices"]
    massLowList = config["massLowList"]
    massUpList = config["massUpList"]
    templatesList = config["templatesList"]
    respFunc = config["respFunc"]
    linSol = config["linSol"]

    # Determine sampled parameters
    if nSlopes == 1:
        alpha = block["parameters", "alpha"]
        alpha1 = alpha
        alpha2 = alpha
    elif nSlopes == 2:
        alpha1 = block["parameters", "alpha1"]
        alpha2 = block["parameters", "alpha2"]
    else:
        raise sys.exit(
            "\nERROR: Number of slopes - parameter not recognized (should be 1 or 2)\n"
        )

    IMFnorms = []
    for tIdx in range(nSSPs):
        parName = "norm" + str(tIdx + 1)
        IMFnorms.append(10 ** (block["parameters", parName]))

    respDex = {"MgDex": 0.0, "CaDex": 0.0, "SiDex": 0.0, "TiDex": 0.0, "NaDex": 0.0}
    if respFunc.abVariations["Mg"]:
        respDex["MgDex"] = block["parameters", "MgDex"]
    if respFunc.abVariations["Ca"]:
        respDex["CaDex"] = block["parameters", "CaDex"]
    if respFunc.abVariations["Si"]:
        respDex["SiDex"] = block["parameters", "SiDex"]
    if respFunc.abVariations["Ti"]:
        respDex["TiDex"] = block["parameters", "TiDex"]
    if respFunc.abVariations["Na"]:
        respDex["NaDex"] = block["parameters", "NaDex"]

    bcov = 10 ** (block["parameters", "logbcov"])

    # Apply response functions
    templatesListNew = copy.deepcopy(templatesList)
    respFunc.applyRespFunctions(templatesListNew, ageIndices, FeHIndices, respDex)

    # Create matrix S and smooth spectra
    S = templatesListNew[0]
    for tIdx in range(nSSPs - 1):
        S = np.concatenate((S, templatesListNew[tIdx + 1]), 1)
    specBasics.smoothSpectra(wavelength, S, sigma)

    # Add additional global covariance parameterized by bcov
    medCov = np.median(cov)
    covNew = cov + bcov * medCov

    # Add local covariance structures, if necessary (in this example, local
    # covariance is included for emission lines OIII and NII)
    bOIII = (0.03 * np.average(flux)) ** 2 / medCov
    bNII = (0.03 * np.average(flux)) ** 2 / medCov
    covOIII = bOIII * (
        specBasics.getGaussianLP(wavelength, 4958.9, 2, medCov)
        + specBasics.getGaussianLP(wavelength, 5006.8, 2.5, medCov)
    )
    covNII = bNII * (
        specBasics.getGaussianLP(wavelength, 6548.1, 2, medCov)
        + specBasics.getGaussianLP(wavelength, 6583.5, 2.5, medCov)
    )
    covNew += covOIII + covNII

    TBIdx1 = np.where(np.logical_and(wavelength > 6860, wavelength < 6920))[0]
    TBIdx2 = np.where(np.logical_and(wavelength > 7160, wavelength < 7360))[0]
    TBIdx3 = np.where(np.logical_and(wavelength > 7580, wavelength < 7680))[0]
    TBIdx4 = np.where(np.logical_and(wavelength > 8130, wavelength < 8360))[0]
    TBIdx = np.concatenate((TBIdx1, TBIdx2, TBIdx3, TBIdx4))
    covNew[TBIdx] *= 9.0

    # Create IMF-prior
    priorList = []
    for tIdx in range(nSSPs):
        prior = SPSbasics.powerlawIMFprior(
            massLowList[tIdx],
            massUpList[tIdx],
            alpha1=alpha1,
            alpha2=alpha2,
            norm=IMFnorms[tIdx],
        )
        priorList.append(prior.weights)

    w0 = priorList[0]
    for tIdx in range(nSSPs - 1):
        w0 = np.concatenate((w0, priorList[tIdx + 1]))

    # Apply polynomial correction
    recSpec = np.dot(S, w0)
    residual = flux / recSpec

    ALT = np.copy(AL)
    ALT = np.transpose(ALT)
    for i in range(nBins):
        ALT[:, i] = ALT[:, i] / covNew[i]

    cL = np.linalg.solve(np.dot(ALT, AL), np.dot(ALT, residual))
    polL = np.dot(AL[:, :], cL).reshape((len(wavelength), 1))
    S = S * polL

    # Update linSol with new templates, covariance and IMF prior
    linSol.updateTemplatesCov(S, covNew)
    linSol.updatePrior(w0)

    # Calculate posterior of sample (prior on cL: to ensure no
    # degeneracy between normalization IMF and polynomial we
    # enforce polynomial with with cL[0] ~ 1.0).
    like = linSol.BayesianEvidence() - (cL[0] - 1.0) ** 2 / (2 * 0.01**2)
    block[section_names.likelihoods, "HBSPS_like"] = like

    return 0


def cleanup(config):
    return 0
