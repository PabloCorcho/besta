import sys
import os
import numpy as np
import h5py
import subprocess
import multiprocessing
from scipy import interpolate
import math
import argparse
import localInt


def main(argv):
	# Parse command line arguments.
	parser = argparse.ArgumentParser(description='Interpolate stellar templates based on MILES library.')
	parser.add_argument('-n', '--nCores', action="store", dest='nCores', type=int, help="use multiprocessing with nCores.")
	parser.add_argument("-b", '--bin', help="bin isochrone spectra to reduce number of templates.", action="store_true")
	args = parser.parse_args()
	
	# Please reference the original papers on which these stellar templates
	# are based if you use them in your work.
	printDisclaimer()
	
	# Determine if multiprocessing should be used
	if args.nCores:
		nCores = args.nCores
	else:
		nCores = 1
	
	# Load MILES data for running interpolator
	global TeffM, loggM, FeHM, SNRM, spectraM, binSpectra
	TeffM, loggM, FeHM, SNRM, spectraM = loadInterpolatorData()
	
	# Determine if isochrone spectra should be binned
	if args.bin:
		binSpectra = True
	else:
		binSpectra = False
	
	# Define age-metallicity grid stellar templates
	logAgeArr = np.linspace(9.175,10.125,39)
	FeHArr = np.linspace(-1.4,0.4,37)
	nAges = len(logAgeArr)
	nFeHs = len(FeHArr)
	
	# Create list with lists that contain filename, age and metallicity 
	# of Parsec isochrones and stitched isochrones
	ParsecList = []
	stitchedList = []
	for FeHIdx in range(nFeHs):
		for ageIdx in range(nAges):
			isochroneFileP = "isochrones/Parsec/parsecIso_logT" + str(logAgeArr[ageIdx]) + "_FeH" + str(FeHArr[FeHIdx]) + ".dat"
			isochroneFileS = "isochrones/stitched/isochrones-S-logT" + str(logAgeArr[ageIdx]) + "-FeH" + str(FeHArr[FeHIdx]) + ".dat"
			ParsecList.append([isochroneFileP, logAgeArr[ageIdx], FeHArr[FeHIdx], 'P'])
			stitchedList.append([isochroneFileS, logAgeArr[ageIdx], FeHArr[FeHIdx], 'S'])
			
	# Interpolate Parsec isochrones
	if nCores == 1:
		for isoIdx in range(len(ParsecList)):
			interpolateIsochroneFile(ParsecList[isoIdx])
	else:
		pool = multiprocessing.Pool(processes=nCores)
		pool.map(interpolateIsochroneFile, ParsecList)
		pool.close()
		pool.join()
		
	# Interpolate stitched isochrones
	if nCores == 1:
		for isoIdx in range(len(stitchedList)):
			interpolateIsochroneFile(stitchedList[isoIdx])
	else:
		pool = multiprocessing.Pool(processes=nCores)
		pool.map(interpolateIsochroneFile, stitchedList)
		pool.close()
		pool.join()

def printDisclaimer():
	print("\nIMPORTANT INFORMATION")
	print("You are about to start the interpolation of the stellar templates")
	print("based on the MILES library and two different sets of isochrones. If")
	print("you use these templates, please reference the following papers on")
	print("which these stellar templates are based:")
	print("* MILES library")
	print("\tSanchez-Blazquez et al., 2006, MNRAS, 371, 703")
	print("* Parsec isochrones")
	print("\tBressan et al., 2012, MNRAS, 427,127")
	print("* stitched isochrones")
	print("\tConroy & van Dokkum, 2012, ApJ, 747, 69")
	print("\tBaraffe et al., 1998, A&A, 337, 403")
	print("\tDotter et al., 2008, ApJS, 178, 89")
	print("\tMarigo et al., 2008, A&A, 482, 883")
	print("\nNote that the interpolation of the stellar templates is quite time")
	print("consuming. It is possible to use multiprocessing with nCores with")
	print("\'python installMILEStemplates.py -n nCores\'.")
	print("\nAlso note that by default the stellar templates are not binned. Use the")
	print("option \'-b\' to apply the binning procedure of Dries et al. (2018).")
	input("\nPress enter to continue")

	
def interpolateIsochroneFile(args):
	"""Interpolate isochrone file.
	
	Use interpolator to create spectra for all stars listed in
	an isochrone file.
	
	Args:
		args[0]: isochroneFile - name of isochrone-file.
		args[1]: log age of the isochrone
		args[2]: metallicity of the isochrone
		args[3]: type of isochrone ('P' = Parsec isochrone, 'S' = stitched isochrone)
		
	"""
	
	# Define isochrone parameters
	isochroneFile = args[0]
	logAgeIso = args[1]
	FeHIso = args[2]
	isoType = args[3]
	
	# Read parameters from isochrone file, define hdf5 file for
	# saving templates and print screen message.
	if isoType == 'P':
		FeH, logAge, mass, logL, logTeff, logg, magB, magV, stage = np.loadtxt(isochroneFile, unpack=True)
		sName = "Parsec/templates-logT" + str(logAgeIso) + "-FeH" + str(FeHIso) + ".hdf5"
		print(("Interpolating Parsec isochrone with log t = " + str(logAgeIso) + " and [Fe/H] = " + str(FeHIso)))
	elif isoType == 'S':
		FeH, logAge, mass, logTeff, logg, logL, magV, magR, isoID, stage = np.loadtxt(isochroneFile, unpack=True)
		sName = "stitched/templates-logT" + str(logAgeIso) + "-FeH" + str(FeHIso) + ".hdf5"
		print(("Interpolating stitched isochrone with log t = " + str(logAgeIso) + " and [Fe/H] = " + str(FeHIso)))
	else:
		raise sys.exit("\nType of isochrone not recognized.\n")
	lum = 10**logL
				
	# Determine number of bins/templates
	nIsoStars = logTeff.shape[0]
	nBins = spectraM.shape[0]
		
	# Convert logTeff to Teff
	Teff = 10**logTeff
	
	# Get mass boundaries isochrone stars.
	massLow, massUp = getMasses(mass, nIsoStars)
	
	# Interpolate spectra.
	spectra = np.empty([nBins, nIsoStars])
	for tIdx in range(nIsoStars):
		spectra[:,tIdx] = localInt.localInterpolator(spectraM, TeffM, loggM, FeHM, SNRM, Teff[tIdx], logg[tIdx], FeH[tIdx])
	
	# Normalize stellar templates on the basis of Johnson V magnitude isochrones
	wavelength = np.linspace(3540.5, 7409.6, 4300)
	RF_V = getResponseFunction(wavelength)
	for tIdx in range(nIsoStars):
		vMag = getMagnitude(spectra[:,tIdx], wavelength, RF_V)
		spectra[:,tIdx] *= 10**( (vMag - magV[tIdx]) / 2.5)
	
	# If required, reduce number of templates with binning
	# procedure described in Dries et al. (2018).
	if binSpectra:
		FeHBinned, logAgeBinned, massBinned, massLowBinned, massUpBinned, lumBinned, spectraBinned = binIsochrone(FeH, logAge, mass, massLow, massUp, lum, stage, spectra, isoType)	
	
	# Save spectra and parameters to hdf5-file.
	if binSpectra:
		saveHDF5file(sName, logAgeBinned, FeHBinned, massBinned, massLowBinned, massUpBinned, lumBinned, spectraBinned)
	else:
		saveHDF5file(sName, logAge, FeH, mass, massLow, massUp, lum, spectra)
	
def loadInterpolatorData():
	"""Load the data that is required for running interpolator.
		
	The interpolator that is used is described in Dries et al. (2016)
	and based on Vazdekis et al. (2003). Note that this version of the
	interpolator does not include the polynomial correction of Dries
	et al. (2016).
	
	"""
	
	# Load hdf5 file
	MILEShdf5 = h5py.File("MILESlibrary.hdf5")
	Teff = np.array(MILEShdf5["Teff"])
	logg = np.array(MILEShdf5["logg"])
	FeH = np.array(MILEShdf5["FeH"])
	SNR = np.array(MILEShdf5["SNR"])
	spectra = np.array(MILEShdf5["spectra"])
	MILEShdf5.close()
	
	# Remove stars with SNR = 0, these stars are considered to be 
	# problematic and are not used by the interpolator
	validStars = np.where(SNR != 0)[0]
	Teff = Teff[validStars]
	logg = logg[validStars]
	FeH = FeH[validStars]
	SNR = SNR[validStars]
	spectra = spectra[:,validStars]
	
	return Teff, logg, FeH, SNR, spectra

def getMasses(mass, nIsoStars):
	"""Determine mass boundaries isochrone stars.
	
	This function determines the mass boundaries between the
	different isochrone stars.
	
	Args:
		mass: initial masses defined by isochrone.
		
	Returns:
		massLow: low-mass boundaries of isochrone stars.
		massUp: high-mass boundaries of isochrone stars.
			
	"""
	massLow = np.empty(nIsoStars)
	massUp = np.empty(nIsoStars)
			
	for id1 in range(nIsoStars):
		if id1 != 0 and mass[id1-1] <= mass[id1]:
			dM1 = (mass[id1] - mass[id1-1]) / 2.0
		else:
			dM1 = (mass[id1+1] - mass[id1]) / 2.0
		massLow[id1] = mass[id1] - dM1
		
		if id1 != nIsoStars-1 and mass[id1] <= mass[id1+1]:
			dM2 = (mass[id1+1] - mass[id1]) / 2.0
		else:
			dM2 = 0
		massUp[id1] = mass[id1] + dM2
	
	return massLow, massUp
	
def getResponseFunction(wavelength):
	"""Determine response function V-filter for given wavelength grid.
		
	Args:
		wavelength: numpy array with wavelength grid.
	
	Returns:
		RF_V: response of V-filter interpolated on 
			wavelength grid.
	
	"""
	nBins = len(wavelength)
	
	# Definition of V-filter
	V_WL = np.linspace(4700, 7000, 47)
	V_F = np.array([0.000, 0.004, 0.032, 0.084, 0.172, 0.310, 0.478, 0.650, 0.802, 0.913, 0.978, 1.000, 0.994, 0.977, 0.950, 0.911, 0.862, 0.806, 0.747, 0.690, 0.634, 0.579, 0.523, 0.467, 0.413, 0.363, 0.317, 0.274, 0.234, 0.200, 0.168, 0.140, 0.114, 0.089, 0.067, 0.050, 0.037, 0.027, 0.020, 0.016, 0.013, 0.012, 0.010, 0.009, 0.007, 0.004, 0.000])
	
	# Spline representation
	tck_V = interpolate.splrep(V_WL, V_F, s = 0)
	
	# Interpolate response function in range of V-filter
	RF_V = np.zeros(nBins)
	for idx in range(nBins):
		if (wavelength[idx] > 4700 and wavelength[idx] < 7000):
			RF_V[idx] = interpolate.splev(wavelength[idx], tck_V)
			
	return RF_V
	
def getMagnitude(spectrum, wavelength, RF_V):
	""" Determine V-magnitude.
	
	Determine V-magnitude of a given spectrum.
	
	Args:
		spectrum: numpy array with flux-values.
		wavelength: numpy array with wavelength grid.
		RF_V: filter function defined on wavelength
			grid
			
	Returns:
		vMag: V-magnitude of input spectrum with respect
			to zeropoint flux defined below.
	
	"""
	
	# Vega zeropoint flux (actual value is not really important, as long
	# as spectra are normalized w.r.t. each other).
	vegaIntV = 363.1 
		
	V_Flux = 0
	normalisation_V = 0
	
	for idx in range(wavelength.shape[0]-1): 
		V_Flux += RF_V[idx] * spectrum[idx] * (wavelength[idx+1] - wavelength[idx])
		normalisation_V += RF_V[idx] * (wavelength[idx+1] - wavelength[idx])
	
	V_Flux /= normalisation_V
	vMag = -2.5*np.log10(V_Flux / vegaIntV)
		
	return vMag
	
def binIsochrone(FeH, logAge, mass, massLow, massUp, lum, stageID, spectra, isoType):
	"""Bin isochrone file.
	
	Bin isochrone hdf5 file under the assumption of a Salpeter IMF,
	as described in Dries et al. (2018)
	
	Args:
		FeH: numpy array with metallicities isochrone stars
		logAge: numpy array with logAge-values isochrone stars
		mass: numpy array with masses isochrone stars
		massLow: numpy array with low-mass boundaries isochrone stars
		massUp: numpy array with high-mass boundaries isochrone stars
		lum: numpy array with luminosities isochrone stars
		stageID: numpy array with integers indicating stellar evolutionary phase
		spectra: 2D numpy array with spectra isochrone stars in columns
		isoType: type of isochrone ('P' = Parsec isochrone, 'S' = stitched isochrone)
	Returns:
		FeHNew: numpy array with binned metallicities
		logAgeNew: numpy array with binned logAge-values
		lumNew: numpy array with binned luminosities
		spectraNew:	2D numpy array with binned spectra
		
	"""
			
	# Determine number of bins and templates
	nBins = spectra.shape[0]
	nIsoStars = spectra.shape[1]
	
	# Get samples of stars that should be combined
	if isoType == 'P':
		samples = getSamplesParsec(nIsoStars, stageID)
	elif isoType == 'S':
		samples = getSamplesStitched(nIsoStars, stageID)
	
	# Determine new number of templates
	newNstars = len(samples)
	
	# Create new mass boundaries
	massLowBinned = np.zeros(newNstars)
	massUpBinned = np.zeros(newNstars)
	for tIdx in range(newNstars):
		massLowBinned[tIdx] = massLow[samples[tIdx][0]]
		massUpBinned[tIdx] = massUp[samples[tIdx][-1]]
	massBinned = massLowBinned + (massUpBinned-massLowBinned)/2.0
	
	# Determine weights spectra based on Salpeter weights
	weightsOr = getPLweights(2.35, massLow, massUp)
	
	# Determine binned spectra and binned parameters
	FeHBinned = np.zeros(newNstars)
	logAgeBinned = np.zeros(newNstars)
	lumBinned = np.zeros(newNstars)
	spectraBinned = np.zeros([nBins,newNstars])
	for tIdx in range(newNstars):
		norm = 0
		FeHBinned[tIdx] = FeH[0]
		logAgeBinned[tIdx] = logAge[0]
		for sIdx in range(len(samples[tIdx])):
			lumBinned[tIdx] += weightsOr[samples[tIdx][sIdx]] * lum[samples[tIdx][sIdx]]
			spectraBinned[:,tIdx] += weightsOr[samples[tIdx][sIdx]] * spectra[:,samples[tIdx][sIdx]]
			norm += weightsOr[samples[tIdx][sIdx]]
		lumBinned[tIdx] /= norm
		spectraBinned[:,tIdx] /= norm
		
	return FeHBinned, logAgeBinned, massBinned, massLowBinned, massUpBinned, lumBinned, spectraBinned
		
def getSamplesParsec(nIsoStars, stageID):
	"""Get samples of Parsec isochrone stars that should be combined into new templates.
	
	Args:
		nIsoStars: the total number of isochrone stars
		stageID: numpy array with integer to indicate evolutionary phase
	Returns:
		samples: list with indices of the templates that should be combined
		
	"""
	
	# Number of templates that are combined for the different evolutionary phases
	MSstep = 2
	GBstep = 3
	CHEBstep = 3
	AGBstep = 8
	
	# Determine which indices of stars belonging to different phases of evolution	
	MSbreak = np.where(stageID >= 2)[0][0]
	GBbreak = np.where(stageID >= 4)[0][0]
	CHEBbreak = np.where(stageID >= 7)[0][0]
	MSidx = []
	GBidx = []
	CHEBidx = []
	AGBidx = []
	for tIdx in range(nIsoStars):
		if tIdx < MSbreak:
			MSidx.append(tIdx)
		elif tIdx < GBbreak:
			GBidx.append(tIdx)
		elif tIdx < CHEBbreak:
			CHEBidx.append(tIdx)
		else:
			AGBidx.append(tIdx)
	
	# Create the samples that should be combined for the different phases of evolution
	samples = []
	nMS = int(math.ceil(float(len(MSidx)) / MSstep))
	nGB = int(math.ceil(float(len(GBidx)) / GBstep))
	nCHEB = int(math.ceil(float(len(CHEBidx)) / CHEBstep))
	nAGB = int(math.ceil(float(len(AGBidx)) / AGBstep))
	
	for tIdx in range(nMS):
		if (tIdx != nMS-1):
			sample = MSidx[tIdx*MSstep:(tIdx+1)*MSstep]
		else:
			sample = MSidx[tIdx*MSstep:]
		samples.append(sample)
	for tIdx in range(nGB):
		if (tIdx != nGB-1):
			sample = GBidx[tIdx*GBstep:(tIdx+1)*GBstep]
		else:
			sample = GBidx[tIdx*GBstep:]
		samples.append(sample)
	for tIdx in range(nCHEB):
		if (tIdx != nCHEB-1):
			sample = CHEBidx[tIdx*CHEBstep:(tIdx+1)*CHEBstep]
		else:
			sample = CHEBidx[tIdx*CHEBstep:]
		samples.append(sample)
	for tIdx in range(nAGB):
		if (tIdx != nAGB-1):
			sample = AGBidx[tIdx*AGBstep:(tIdx+1)*AGBstep]
		else:
			sample = AGBidx[tIdx*AGBstep:]
		samples.append(sample)
	
	return samples
	
def getSamplesStitched(nIsoStars, stageID):
	"""Get samples of stitched isochrone stars that should be combined into new templates.
	
	Note that binning is slightly different here, and we distinguish the following stages:
	* stage 1: mass templates < 0.25 Msun
	* stage 2: main sequence >= 0.25 Msun
	* stage 3: RGB and above for Dartmouth isochones
	* stage 4: Padova isochrones
	
	Args:
		nIsoStars: the total number of isochrone stars
		stageID: numpy array with integer to indicate evolutionary phase
	Returns:
		samples: list with indices of the templates that should be combined
		
	"""
	
	# Number of templates that are combined for the different evolutionary phases
	ST1step = 1
	ST2step = 3
	ST3step = 5
	ST4step = 8
	
	# Determine which indices of stars belonging to different phases of evolution		
	break1 = np.where(stageID >= 2)[0][0]
	break2 = np.where(stageID >= 3)[0][0]
	break3 = np.where(stageID >= 4)[0][0]
	ST1idx = []
	ST2idx = []
	ST3idx = []
	ST4idx = []
	for tIdx in range(nIsoStars):
		if tIdx < break1:
			ST1idx.append(tIdx)
		elif tIdx < break2:
			ST2idx.append(tIdx)
		elif tIdx < break3:
			ST3idx.append(tIdx)
		else:
			ST4idx.append(tIdx)
	
	# Create the samples that should be combined for the different phases of evolution
	samples = []
	n1 = int(math.ceil(float(len(ST1idx)) / ST1step))
	n2 = int(math.ceil(float(len(ST2idx)) / ST2step))
	n3 = int(math.ceil(float(len(ST3idx)) / ST3step))
	n4 = int(math.ceil(float(len(ST4idx)) / ST4step))
	
	for tIdx in range(n1):
		if (tIdx != n1-1):
			sample = ST1idx[tIdx*ST1step:(tIdx+1)*ST1step]
		else:
			sample = ST1idx[tIdx*ST1step:]
		samples.append(sample)
	for tIdx in range(n2):
		if (tIdx != n2-1):
			sample = ST2idx[tIdx*ST2step:(tIdx+1)*ST2step]
		else:
			sample = ST2idx[tIdx*ST2step:]
		samples.append(sample)
	for tIdx in range(n3):
		if (tIdx != n3-1):
			sample = ST3idx[tIdx*ST3step:(tIdx+1)*ST3step]
		else:
			sample = ST3idx[tIdx*ST3step:]
		samples.append(sample)
	for tIdx in range(n4):
		if (tIdx != n4-1):
			sample = ST4idx[tIdx*ST4step:(tIdx+1)*ST4step]
		else:
			sample = ST4idx[tIdx*ST4step:]
		samples.append(sample)
	
	return samples
	
def getPLweights(alpha, massLow, massUp):
	"""Determine weights spectra based on power law IMF.
	
	Args:
		alpha: slope of the IMF
		massLow: low-mass boundaries templates
		massUp: high-mass boundaries templates
	Returns:
		weights: weights spectra based on power law IMF
		
	"""
	nStars = len(massLow)
	weights = np.zeros(nStars)
	for tIdx in range(nStars):
		weights[tIdx] = ( (massLow[tIdx])**(-alpha+1) - (massUp[tIdx])**(-alpha+1) )
		
	return weights
	
def saveHDF5file(saveFile, logAge, FeH, mass, massLow, massUp, lum, spectra):
	f = h5py.File(saveFile, "w")
	f.create_dataset("logAge", data=logAge)
	f.create_dataset("FeH", data=FeH)
	f.create_dataset("mass", data=mass)
	f.create_dataset("massLow", data=massLow)
	f.create_dataset("massUp", data=massUp)
	f.create_dataset("luminosity", data=lum)
	f.create_dataset("spectra", data=spectra)
	f.close()

if (__name__ == "__main__"):
	main(sys.argv[1:])