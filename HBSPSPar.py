import sys
import numpy as np
import cosmosis
from cosmosis.datablock import option_section, names as section_names
import specBasics
import SPSbasics

def X2min(spectrum, recSp, cov):
	# Determine residual, divide first residual vector by 
	# diagonal elements covariance matrix.
	residual1 = recSp - spectrum
	residual2 = np.copy(residual1)
	residual1 /= cov
		
	# Determine likelihood term (i.e. X2-value)
	chiSq = -0.5*np.dot(residual1, residual2)
	
	return chiSq
	
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
	bcov = 10**(options[option_section, "logbcov"])
	polOrder = options[option_section, "polOrder"]
	abVariations = dict()
	abVariations['Mg'] = options[option_section, "sampleMg"]
	abVariations['Ca'] = options[option_section, "sampleCa"]
	abVariations['Si'] = options[option_section, "sampleSi"]
	abVariations['Ti'] = options[option_section, "sampleTi"]
	abVariations['Na'] = options[option_section, "sampleNa"]
	if any(value == True for value in list(abVariations.values())):
		resFunHDF5 = options[option_section, "resFunHDF5"]
	else:
		resFunHDF5 = None
	
	# Read wavelength, SSP spectrum and create covariance matrix
	wavelength, flux, error = np.loadtxt(fileName, unpack=True)
	nBins = len(wavelength)
	cov = error*error
	
	# Select wavelength range that you want to use
	goodIdx = np.where(wavelength <= 8800)[0]
	wavelength = wavelength[goodIdx]
	flux = flux[goodIdx]
	cov = cov[goodIdx]
	error = error[goodIdx]
	nBins = len(wavelength)
	
	# Add additional global covariance parameterized by bcov
	medCov = np.median(cov)
	cov = cov + bcov*medCov
	
	# Add local covariance structures, if necessary (in this example, local
	# covariance is included for emission lines OIII and NII)
	bOIII = (0.03*np.average(flux))**2 / medCov
	bNII = (0.03*np.average(flux))**2 / medCov
	covOIII = bOIII*(specBasics.getGaussianLP(wavelength, 4958.9, 2, medCov) + specBasics.getGaussianLP(wavelength, 5006.8, 2.5, medCov))
	covNII = bNII*(specBasics.getGaussianLP(wavelength, 6548.1, 2, medCov) + specBasics.getGaussianLP(wavelength, 6583.5, 2.5, medCov))
	cov += (covOIII + covNII)
	
	# Set available ages and metallicities.
	logAgeArr = np.linspace(9.175,10.125,39)
	FeHArr = np.linspace(-1.4,0.4,37)
	nAges = len(logAgeArr)
	nFeHs = len(FeHArr)
	
	# Create stTemplates-object for quickly reading SSP spectra.
	stTemplates = SPSbasics.stellarTemplates(templatesDir, logAgeArr, FeHArr)
	
	# Load response functions.	
	respFunc = SPSbasics.respFunctions(nSSPs, abVariations, wavelength, goodIdx, resFunHDF5)
	
	# Basis of Legendre polynomials for multiplicative polynomial
	AL = specBasics.getLegendrePolynomial(wavelength, polOrder, bounds=None)
	
	# Determine average flux for normalization IMF
	avFlux = np.average(flux)
	
	# Pass parameters to execute function.
	config = {}
	config['flux'] = flux
	config['avFlux'] = avFlux
	config['wavelength'] = wavelength
	config['nBins'] = nBins
	config['goodIdx'] = goodIdx
	config['error'] = error
	config['cov'] = cov
	config['nSSPs'] = nSSPs
	config['nSlopes'] = nSlopes
	config['polOrder'] = polOrder
	config['AL'] = AL
	config['logAgeArr'] = logAgeArr
	config['FeHArr'] = FeHArr
	config['bcov'] = bcov
	config['stTemplates'] = stTemplates
	config['respFunc'] = respFunc

	return config
	
def execute(block, config):
	"""Function executed by sampler
	This is the function that is executed many times by the sampler. The
	likelihood resulting from this function is the evidence on the basis
	of which the parameter space is sampled.
	"""
	
	# Obtain parameters from setup
	flux = config['flux']
	avFlux = config['avFlux']
	wavelength = config['wavelength']
	nBins = config['nBins']
	goodIdx = config['goodIdx']
	error = config['error']
	cov = config['cov']
	nSSPs = config['nSSPs']
	nSlopes = config['nSlopes']
	polOrder = config['polOrder']
	AL = config['AL']
	logAgeArr = config['logAgeArr']
	FeHArr = config['FeHArr']
	bcov = config['bcov']
	stTemplates = config['stTemplates']
	respFunc = config['respFunc']
				
	# Determine sampled parameters
	if nSlopes == 1:
		alpha = block["parameters", "alpha"]
		alpha1 = alpha
		alpha2 = alpha
	elif nSlopes == 2:
		alpha1 = block["parameters", "alpha1"]
		alpha2 = block["parameters", "alpha2"]
	else:
		raise sys.exit("\nERROR: Number of slopes - parameter not recognized (should be 1 or 2)\n")
	
	sigma = np.sqrt(block["parameters", "sigma"]**2 - 100.0**2)	# Stellar templates already smoothed to 100 km/s.
	
	ages = []
	FeHs = []
	for tIdx in range(nSSPs):
		parAge = "age" + str(tIdx+1)
		parFeH = "FeH" + str(tIdx+1)
		ages.append(block["parameters", parAge])
		FeHs.append(block["parameters", parFeH])
	
	lumFrs = []
	for tIdx in range(nSSPs-1):
		parLumFr = "lumFr" + str(tIdx+1)
		lumFrs.append(block["parameters", parLumFr])
		
	respDex = {'MgDex':0.0, 'CaDex':0.0, 'SiDex':0.0, 'TiDex':0.0, 'NaDex':0.0}
	if respFunc.abVariations['Mg']:
		respDex['MgDex'] = block["parameters", "MgDex"]
	if respFunc.abVariations['Ca']:
		respDex['CaDex'] = block["parameters", "CaDex"]
	if respFunc.abVariations['Si']:
		respDex['SiDex'] = block["parameters", "SiDex"]
	if respFunc.abVariations['Ti']:
		respDex['TiDex'] = block["parameters", "TiDex"]
	if respFunc.abVariations['Na']:
		respDex['NaDex'] = block["parameters", "NaDex"]
		
	# The total sum of luminosity fractions should always be one. If
	# the sum is larger than one, then penalize model with additional
	# (strict) prior to prevent degeneracies with multiplicative
	# polynomial. Set the fraction of the last remaining SSP to one
	# minus the other luminosity fractions (or zero if this is already
	# higher than one). This saves one parameter in sampling procedure.
	sumLumFrs = sum(lumFrs)
	if sum(lumFrs) > 1.0:
		prLumFrs = (sumLumFrs-1.0)**2 / (2* 0.001**2)
		lumFrs.append(0.0)
	else:
		prLumFrs = 0.0
		lumFrs.append(1.0-sumLumFrs)
	
	# Convert sampled values age-metallicity into discrete indices
	FeHIndices = np.zeros(nSSPs, dtype=int)
	ageIndices = np.zeros(nSSPs, dtype=int)
	ages = np.asarray(ages)
	logAges = np.log10(1e9*ages)
	for tIdx in range(nSSPs):
		ageIndices[tIdx] = (np.abs(logAgeArr - logAges[tIdx])).argmin()
		FeHIndices[tIdx] = (np.abs(FeHArr - FeHs[tIdx])).argmin()
	
	# Load stellar templates data
	massLowList = []
	massUpList = []
	templatesList = []
	for tIdx in range(nSSPs):
		massLow, massUp, spectra, lum = stTemplates.getTemplatesIdx(ageIndices[tIdx], FeHIndices[tIdx])
		spectra = spectra[goodIdx,:]
		
		massLowList.append(massLow)
		massUpList.append(massUp)		
		templatesList.append(spectra)
		
	# Apply response functions
	respFunc.applyRespFunctions(templatesList, ageIndices, FeHIndices, respDex)
		
	# Create (unnormalized) IMF-prior
	priorList = []
	for tIdx in range(nSSPs):
		prior = SPSbasics.powerlawIMFprior(massLowList[tIdx], massUpList[tIdx], alpha1=alpha1, alpha2=alpha2, norm=1.0)
		priorList.append(prior.weights)
		
	# Smooth spectra, calculate SSP spectra and combine SSPs into CSP
	recSpecCSP = np.zeros(nBins)
	for tIdx in range(nSSPs):
		templatesList[tIdx] = specBasics.smoothSpectra(wavelength, templatesList[tIdx], sigma)
		recSpecSSP = np.dot(templatesList[tIdx], priorList[tIdx])
		normSSP = avFlux / np.average(recSpecSSP)
		recSpecCSP += lumFrs[tIdx]*normSSP*recSpecSSP
	
	# Apply polynomial correction to reconstructed spectrum
	residual = flux / recSpecCSP

	ALT = np.copy(AL)
	ALT = np.transpose(ALT)
	for i in range(nBins):
		ALT[:,i] = ALT[:,i] / cov[i]
		
	cL = np.linalg.solve(np.dot(ALT,AL), np.dot(ALT, residual))
	polL = np.dot(AL,cL)
	recSpecCSP *= polL
	
	# Calculate likelihood-value of the fit
	like = X2min(flux, recSpecCSP, cov)
	
	# Final posterior for sampling: combination likelihood and prior lumFrs
	block[section_names.likelihoods, "HBSPSPar_like"] = like - prLumFrs

	return 0
	
def cleanup(config):
		
	return 0
