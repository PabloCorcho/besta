""" Local interpolator for stellar spectra

This interpolator is based on the algorithm of Vazekis et al. (2003)
and described in Dries et al. (2016). For a given combination of stellar
parameters theta, logg and Fe/H the algorithm selects nearby stars in this
three-dimensional space and applies a weighting scheme to calculate an 
interpolated spectrum

"""
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
import scipy.optimize

def countStars(theta, logg, FeH, theta0, logg0, FeH0, dTheta, dLogg, dFeH, nStars):
	"""Count stars in a given 3D box.
	
	This function counts the numbers of stars in the library in a box centered 
	at {theta0, logg0, FeH0} and with dimensions 2*dTheta x 2*dLogg x 2*dFeH.
	theta, logg and FeH are numpy arrays with the parameters of the stars in 
	the library.
	
	For Teff < 3500 and Teff > 9000, the metallicity is not taken into account.
	
	"""
	
	countedStars = 0
	
	if (5040.0 / theta0 < 3500.0 or 5040.0 / theta0 > 9000.0):
		for sIdx in range(nStars):
			if (abs(theta[sIdx] - theta0) < dTheta) and (abs(logg[sIdx] - logg0) < dLogg):
				countedStars += 1
	else:
		for sIdx in range(nStars):
			if (abs(theta[sIdx] - theta0) < dTheta) and (abs(logg[sIdx] - logg0) < dLogg) and (abs(FeH[sIdx] - FeH0) < dFeH):
				countedStars += 1
	
	return countedStars

def getDensity(theta, logg, FeH, theta0, logg0, FeH0, dTheta, dLogg, dFeH, nStars):
	"""Determine local density of stars.
	
	Determine the local density of stars that is used to determine standard
	deviation of theta, log g and [Fe/H] in the interpolator. Initial size
	of the box is 3*dThetam x 3*dLoggm x 3*dFeHm, if no stars are found in that
	box it is enlarged in all three directions with one (minimum) standard 
	deviation.
	
	Note that in this case the metallicity is always taken into account 
	because otherwise the definition of density is not consistent.
	
	"""
	
	countedStars = 0
	nIterations = 2
	
	while (countedStars == 0):
		nIterations += 1
		countedStars = countStars(theta, logg, FeH, theta0, logg0, FeH0, nIterations*dTheta, nIterations*dLogg, nIterations*dFeH, nStars)
	
	density = float(countedStars) / (2*nIterations*dTheta * 2*nIterations*dLogg *  2*nIterations*dFeH)

	return density
	
def getParameterSTDs(theta, logg, FeH, Teff0, theta0, logg0, FeH0, nStars, maxDensity, SNmax):
	""" Determine uncertainties stellar parameters as in Vazdekis et al. (2003).
	
	IMPORTANT: the min/max uncertainty values that are used here are the ones 
	from Vazdekis et al. (2003), change these if necessary. 
	
	"""
	
	# Define minimum/maximum uncertainty in stellar parameters
	#
	dThetam = 0.009
	dThetaM = 0.1696
	dLoggm = 0.18
	dFeHm = 0.09
	dLoggM = 0.511
	dFeHM = 0.408
	
	# Ensure that dTeffm is at least 60 K
	Teffnew = 5040.0 / (theta0 + 0.009)
	if (abs(Teffnew- Teff0) < 60.0): 
		dThetam = abs((5040 / (Teff0-60)) - theta0)
	
	# Ensure that dTeffM is not more than 3350 K
	Teffnew = 5040.0 / (theta0 - 0.1696)
	if (abs(Teffnew - Teff0) > 3350): #
		dThetaM = abs((5040 / (Teff0+3350)) - theta0)

	# Determine density of stars at requested point
	density = getDensity(theta, logg, FeH, theta0, logg0, FeH0, dThetam, dLoggm, dFeHm, nStars)
	
	# Determine uncertainties stellar parameters at the requested point
	# and based on local density of stars
	if (density > maxDensity):
		dTheta = dThetam
		dLogg = dLoggm
		dFeH = dFeHm
	else:
		dTheta = dThetam * np.exp( ((density-maxDensity)/maxDensity)**2 * np.log(dThetaM / dThetam) )
		dLogg = dLoggm * np.exp( ((density-maxDensity)/maxDensity)**2 * np.log(dLoggM / dLoggm) )
		dFeH = dFeHm * np.exp( ((density-maxDensity)/maxDensity)**2 * np.log(dFeHM / dFeHm) )
		
	return dTheta, dLogg, dFeH
	
	
def findBoxStars(theta0, logg0, FeH0, theta, logg, FeH, thetam, loggm, FeHm, thetaM, loggM, FeHM, nStars):
	"""Find stars in parameter box.
	
	Function to determine which stars lie within a specified range of theta0,
	logg0 and FeH0. These stars are returned as a list.
	
	For Teff0 < 3500 and Teff > 9000 metallicity is ignored.
	
	"""
	
	boxStars = []
	
	if (5040.0 / theta0 < 3500.0 or 5040.0 / theta0 > 9000.0):
		for id1 in range(nStars):
			if (thetam <= theta[id1] < thetaM) and (loggm <= logg[id1] < loggM):
				boxStars.append(id1)
	else:
		for id1 in range(nStars):
			if (thetam <= theta[id1] < thetaM) and (loggm <= logg[id1] < loggM) and (FeHm <= FeH[id1] < FeHM):
				boxStars.append(id1)
	
	return boxStars

def getBoxStars(theta, logg, FeH, theta0, logg0, FeH0, dTheta, dLogg, dFeH, nStars):
	""" Find stars in eight boxes that surround requested point {theta0, logg0, FeH0}.
	
	"""
	
	boxStars = [[],[],[],[],[],[],[],[]]
	beginCount = 4
	maxCount = 30 # max size of box in terms of standard deviation parameters (change if necessary)
	
	boxStars[0] = findBoxStars(theta0, logg0, FeH0, theta, logg, FeH, theta0, logg0, FeH0, theta0+3*dTheta, logg0+3*dLogg, FeH0+3*dFeH, nStars)
	counter = beginCount
	
	while (len(boxStars[0]) == 0 and counter <= maxCount):
		boxStars[0] = findBoxStars(theta0, logg0, FeH0, theta, logg, FeH, theta0, logg0, FeH0, theta0+counter*dTheta, logg0+counter*dLogg, FeH0+counter*dFeH, nStars)
		counter += 1

	boxStars[1] = findBoxStars(theta0, logg0, FeH0, theta, logg, FeH, theta0, logg0, FeH0-3*dFeH, theta0+3*dTheta, logg0+3*dLogg, FeH0, nStars)
	counter = beginCount
	while (len(boxStars[1]) == 0 and counter <= maxCount):
		boxStars[1] = findBoxStars(theta0, logg0, FeH0, theta, logg, FeH, theta0, logg0, FeH0-counter*dFeH, theta0+counter*dTheta, logg0+counter*dLogg, FeH0, nStars)
		counter += 1
	
	boxStars[2] = findBoxStars(theta0, logg0, FeH0, theta, logg, FeH, theta0, logg0-3*dLogg, FeH0, theta0+3*dTheta, logg0, FeH0+3*dFeH, nStars)
	counter = beginCount
	while (len(boxStars[2]) == 0 and counter <= maxCount):
		boxStars[2] = findBoxStars(theta0, logg0, FeH0, theta, logg, FeH, theta0, logg0-counter*dLogg, FeH0, theta0+counter*dTheta, logg0, FeH0+counter*dFeH, nStars)
		counter += 1
	
	boxStars[3] = findBoxStars(theta0, logg0, FeH0, theta, logg, FeH, theta0, logg0-3*dLogg, FeH0-3*dFeH, theta0+3*dTheta, logg0, FeH0, nStars)
	counter = beginCount
	while (len(boxStars[3]) == 0 and counter <= maxCount):
		boxStars[3] = findBoxStars(theta0, logg0, FeH0, theta, logg, FeH, theta0, logg0-counter*dLogg, FeH0-counter*dFeH, theta0+counter*dTheta, logg0, FeH0, nStars)
		counter += 1
	
	boxStars[4] = findBoxStars(theta0, logg0, FeH0, theta, logg, FeH, theta0-3*dTheta, logg0, FeH0, theta0, logg0+3*dLogg, FeH0+3*dFeH, nStars)
	counter = beginCount
	while (len(boxStars[4]) == 0 and counter <= maxCount):
		boxStars[4] = findBoxStars(theta0, logg0, FeH0, theta, logg, FeH, theta0-counter*dTheta, logg0, FeH0, theta0, logg0+counter*dLogg, FeH0+counter*dFeH, nStars)
		counter += 1
	
	boxStars[5] = findBoxStars(theta0, logg0, FeH0, theta, logg, FeH, theta0-3*dTheta, logg0, FeH0-3*dFeH, theta0, logg0+3*dLogg, FeH0, nStars)
	counter = beginCount
	while (len(boxStars[5]) == 0 and counter <= maxCount):
		boxStars[5] = findBoxStars(theta0, logg0, FeH0, theta, logg, FeH, theta0-counter*dTheta, logg0, FeH0-counter*dFeH, theta0, logg0+counter*dLogg, FeH0, nStars)
		counter += 1
	
	boxStars[6] = findBoxStars(theta0, logg0, FeH0, theta, logg, FeH, theta0-3*dTheta, logg0-3*dLogg, FeH0, theta0, logg0, FeH0+3*dFeH, nStars)
	counter = beginCount
	while (len(boxStars[6]) == 0 and counter <= maxCount):
		boxStars[6] = findBoxStars(theta0, logg0, FeH0, theta, logg, FeH, theta0-counter*dTheta, logg0-counter*dLogg, FeH0, theta0, logg0, FeH0+counter*dFeH, nStars)
		counter += 1

	boxStars[7] = findBoxStars(theta0, logg0, FeH0, theta, logg, FeH, theta0-3*dTheta, logg0-3*dLogg, FeH0-3*dFeH, theta0, logg0, FeH0, nStars)
	counter = beginCount
	while (len(boxStars[7]) == 0 and counter <= maxCount):
		boxStars[7] = findBoxStars(theta0, logg0, FeH0, theta, logg, FeH, theta0-counter*dTheta, logg0-counter*dLogg, FeH0-counter*dFeH, theta0, logg0, FeH0, nStars)
		counter += 1
		
	return boxStars

def getBoxSpectra(theta, logg, FeH, theta0, logg0, FeH0, dTheta, dLogg, dFeH, boxStars, nStars, nBins, spectra, SN, SNmax):
	"""Determine weighted box spectrum
	
	For each of the boxes in which stars were found, this function determines a representive
	spectrum. See Vazdekis et al. (2003) and Dries et al. 2016 for the weighting scheme. In 
	addition, this function determines for each box a set of weighted box parameters (same
	weighting scheme as for the spectra).
		
	Args:
		theta, logg, FeH: stellar parameters library stars
		theta0, logg0, FeH0: requested interpolation point
		dTheta, dLogg, dFeH: adopted uncertainties stellar parameters
		boxStars: list with lists of stars in different boxes
		nBins: number of data points in spectrum
		nStars: number of stars in library
		spectra: numpy array with library spectra (nbins x nStars), so each column = 1 spectrum
		SN: SN ratios of the library stars
		SNmax: cutoff of SN weighting, above SNmax all stars get SN-weight 1
	
	"""
		
	if (5040.0 / theta0 < 3500.0 or 5040.0 / theta0 > 9000.0):
		boxSpectra = [[],[],[],[]]
		boxParameters = np.zeros([4,2])
		for bIdx in range(4):
			nBoxStars = len(boxStars[bIdx])
			if (nBoxStars != 0):
				weights = np.zeros(nBoxStars)
				boxSpectra[bIdx] = np.zeros(nBins)
				for sIdx in range(nBoxStars):
					SNweight = SN[boxStars[bIdx][sIdx]]**2 / SNmax**2
					if (SN[boxStars[bIdx][sIdx]] > SNmax):
						SNweight = 1.0
					weights[sIdx] = np.exp( -( (theta[boxStars[bIdx][sIdx]] - theta0) / dTheta )**2 ) * np.exp( -( (logg[boxStars[bIdx][sIdx]] - logg0) / dLogg )**2 ) * SNweight
					boxSpectra[bIdx] += weights[sIdx] * spectra[:,boxStars[bIdx][sIdx]]
					boxParameters[bIdx][0] += weights[sIdx] * theta[boxStars[bIdx][sIdx]]
					boxParameters[bIdx][1] += weights[sIdx] * logg[boxStars[bIdx][sIdx]]
				normalization = np.sum(weights)
				boxSpectra[bIdx] /= normalization
				boxParameters[bIdx][0] /= normalization
				boxParameters[bIdx][1] /= normalization
	else:
		boxSpectra = [[],[],[],[],[],[],[],[]]
		boxParameters = np.zeros([8,3])
		for bIdx in range(8):
			nBoxStars = len(boxStars[bIdx])
			if (nBoxStars != 0):
				weights = np.zeros(nBoxStars)
				boxSpectra[bIdx] = np.zeros(nBins)
				for sIdx in range(nBoxStars):
					SNweight = SN[boxStars[bIdx][sIdx]]**2 / SNmax**2
					if (SN[boxStars[bIdx][sIdx]] > SNmax):
						SNweight = 1.0
					weights[sIdx] = np.exp( -( (theta[boxStars[bIdx][sIdx]] - theta0) / dTheta )**2 ) * np.exp( -( (logg[boxStars[bIdx][sIdx]] - logg0) / dLogg )**2 ) * np.exp( -( (FeH[boxStars[bIdx][sIdx]] - FeH0) / dFeH )**2 ) * SNweight
					boxSpectra[bIdx] += weights[sIdx] * spectra[:,boxStars[bIdx][sIdx]]
					boxParameters[bIdx][0] += weights[sIdx] * theta[boxStars[bIdx][sIdx]]
					boxParameters[bIdx][1] += weights[sIdx] * logg[boxStars[bIdx][sIdx]]
					boxParameters[bIdx][2] += weights[sIdx] * FeH[boxStars[bIdx][sIdx]]
				normalization = np.sum(weights)
				boxSpectra[bIdx] /= normalization
				boxParameters[bIdx][0] /= normalization
				boxParameters[bIdx][1] /= normalization
				boxParameters[bIdx][2] /= normalization
			
	return boxSpectra, boxParameters

def getBoxWeights(boxParameters, theta0, logg0, FeH0, dTheta, dLogg, dFeH):
	"""Determine weights of different boxes surrounding requested point {theta0, logg0, FeH0}.
	
	This function determines the weight of the different boxes surrounding the requested
	interpolation point. Note that the stars in these boxes have already been weighted
	and a corresponding set of box-parameters has been calculated. These weighted 
	box-parameters are used to weight the different boxes.
	
	"""

	if (5040.0 / theta0 < 3500.0 or 5040.0 / theta0 > 9000.0):
		boxWeights = np.zeros(4)
		for id1 in range(4):
			if (boxParameters[id1][0] != 0):
				boxWeights[id1] = np.exp( -( (boxParameters[id1][0] - theta0) / dTheta )**2 ) * np.exp( -( (boxParameters[id1][1] - logg0) / dLogg )**2 )
	else:
		boxWeights = np.zeros(8)
		for id1 in range(8):
			if (boxParameters[id1][0] != 0):
				boxWeights[id1] = np.exp( -( (boxParameters[id1][0] - theta0) / dTheta )**2 ) * np.exp( -( (boxParameters[id1][1] - logg0) / dLogg )**2 ) * np.exp( -( (boxParameters[id1][2] - FeH0) / dFeH )**2 )
	
	return boxWeights

def localInterpolator(spectra, Teff, logg, FeH, SN, Teff0, logg0, FeH0):
	"""Interpolate a stellar spectrum.
	
	This is the main function that one can call from an external script to interpolate
	a stellar spectrum.
	
	Args:
		spectra: numpy array with the spectra that has shape nBins x nStars
		Teff: numpy array with the effective temperatures of the stars in
			the library (not theta, theta is only used internally!)
		logg: numpy array with the surface gravity of the library stars
		FeH: numpy array with the metallicities of the library stars
		SN: numpy array with the SN-ratios of the library stars
		Teff0, logg0, FeH: parameters of the interpolation point
	
		IMPORTANT: values of maxDensity and SNmax are derived for the
		MILES library as it is used in Dries et al. 2016, change these
		values if you use the interpolator with a different library.
		
	"""
	
	# Values for the MILES library as used in Dries et al. (2016),
	# maxDensity is based on 97.5% percentile of densities as measured
	# at locations of all stars in the library and SNmax is based on 
	# 95% percentile of SN-ratios of all stars in the library. CHANGE
	# THESE VALUES IF YOU USE A DIFFERENT LIBRARY.
	maxDensity = 314 
	SNmax = 204 
	
	# Convert effective temperatures to theta
	theta = 5040.0 / Teff
	theta0 = 5040.0 / Teff0
	
	# Determine number of stars in library and number of bins spectrum
	nStars = Teff.shape[0]	
	nBins = spectra.shape[0]
	
	# Determine uncertainties stellar parameters
	dTheta, dLogg, dFeH = getParameterSTDs(theta, logg, FeH, Teff0, theta0, logg0, FeH0, nStars, maxDensity, SNmax)
	
	# Get stars in different boxes surroundig requested point, for 
	# Teff < 3500 and Teff > 9000 K reduce number of boxes to 4.
	boxStars = getBoxStars(theta, logg, FeH, theta0, logg0, FeH0, dTheta/2.0, dLogg/2.0, dFeH/2.0, nStars)
	
	if (5040.0 / theta0 < 3500.0 or 5040.0 / theta0 > 9000.0):
		boxStars = [boxStars[0], boxStars[2], boxStars[4], boxStars[6]]
 	 
 	# Determined weighted spectra, parameters and SN-ratios of boxes
 	boxSpectra, boxParameters = getBoxSpectra(theta, logg, FeH, theta0, logg0, FeH0, dTheta, dLogg, dFeH, boxStars, nStars, nBins, spectra, SN, SNmax)
 	
 	# Determine weights different boxes
 	boxWeights = getBoxWeights(boxParameters, theta0, logg0, FeH0, dTheta, dLogg, dFeH)
 	
	# Calculate interpolated spectrum
	recSpectrum = np.zeros(nBins)
 	normalization = 0
 	
 	if (5040.0 / theta0 < 3500.0 or 5040.0 / theta0 > 9000.0):
		for id1 in range(4):
			if boxWeights[id1] != 0:
				recSpectrum += boxWeights[id1] * boxSpectra[id1]
				normalization += boxWeights[id1]
				
	else:
 		for id1 in range(8):
			if boxWeights[id1] != 0:
				recSpectrum += boxWeights[id1] * boxSpectra[id1]
				normalization += boxWeights[id1]
	
	if normalization != 0:
		recSpectrum /= normalization
	else:
		print("\n\nNormalization of the interpolated spectrum is found to be 0. This")
		print("probably implies that no stars have been found around the requested")
		print("point and that the point is out of range for the intepolator.\n")
		raise sys.exit()
	
	return recSpectrum