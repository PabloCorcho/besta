"""SPS module.

This module contains classes and functions related to
Stellar Population Synthesis. 

"""
import sys
import os.path
import numpy as np
import h5py
import hbsps.specBasics as specBasics

class stellarTemplates:
	"""Class for reading stellar templates data.
	
	This class is used for reading stellar templates data. For each
	age-metallicity gridpoint there is one hdf5-file with the spectral
	data of the corresponding SSP. 
	
	Args:
		templatesDir: name of directory with stellar templates.
		logAgeArr: numpy array with logAge-values of age-metallicity grid.
		FeHArr: numpy array with FeH-values of age-metallicity grid.
		
	"""
	
	def __init__(self, templatesDir, logAgeArr, FeHArr):
		self.logAgeArr = logAgeArr
		self.FeHArr = FeHArr
		self.nAges = len(self.logAgeArr)
		self.nFeHs = len(self.FeHArr)
		self.hdf5Files = []
		for ageIdx in range(self.nAges):
			for FeHIdx in range(self.nFeHs):
				self.hdf5Files.append(templatesDir + "templates-logT" + str(self.logAgeArr[ageIdx]) + "-FeH" + str(self.FeHArr[FeHIdx]) + ".hdf5")

	def __readTemplates(self, ageIdx, FeHIdx):
		"""Read hdf5-file for a given age-metallicity combination.
		
		"""
		
		# Open hdf5 file
		fileIdx = ageIdx*self.nFeHs + FeHIdx
		fName = self.hdf5Files[fileIdx]
		if not os.path.isfile(fName):
			raise sys.exit("\nERROR: HDF5-file \'" + fName + "\' does not exist.\n")
		try:
			hdf5File = h5py.File(fName)
		except IOError:
			raise sys.exit("\nERROR: HDF5-file \'" + fName + "\' is not a valid HDF5-file.\n")
						
		# Read data sets from hdf5 file
		try:
			mass = np.array(hdf5File["mass"])
			massLow = np.array(hdf5File["massLow"])
			massUp = np.array(hdf5File["massUp"])
			spectra = np.array(hdf5File["spectra"])
			lum = np.array(hdf5File["luminosity"])
		except KeyError:
			raise sys.exit("\nERROR: One of the required datasets is missing in HDF5-file \'" + fName + "\'.\n")
	
		# Close hdf5 file
		hdf5File.close()
		
		return mass, massLow, massUp, spectra, lum
		
	def __mapAgeMetallicity(self, age, FeH):
		"""Map age-metallicity combination to closest gridpoint.
		
		"""
		logAge = np.log10(1e9*age)
		ageIdx = (np.abs(self.logAgeArr - logAge)).argmin()
		FeHIdx = (np.abs(self.FeHArr - FeH)).argmin()
		
		return ageIdx, FeHIdx
		
	def getTemplatesIdx(self, ageIdx, FeHIdx):
		"""Read stellar templates data of SSP by index.
		
		Args:
			ageIdx: Index of required age in logAgeArr.
			FeHIdx: Index of required FeH in FeHArr.
			
		Returns:
			massLow: low-mass boundaries of stellar templates
			massUp: upper-mass boundaries of stellar templates
			spectra: spectra of stellar templates
			lum: luminosities of stellar templates
		
		"""
		mass, massLow, massUp, spectra, lum = self.__readTemplates(ageIdx, FeHIdx)
		
		return massLow, massUp, spectra, lum
		
	def getTemplates(self, age, FeH):
		"""Read stellar templates data of SSP by index.
		
		Args:
			age: Required age in Gyr.
			FeH: Required [Fe/H].
			
		Returns:
			massLow: low-mass boundaries of stellar templates
			massUp: upper-mass boundaries of stellar templates
			spectra: spectra of stellar templates
			lum: luminosities of stellar templates
		
		"""
		ageIdx, FeHIdx = self.__mapAgeMetallicity(age, FeH)
		mass, massLow, massUp, spectra, lum = self.__readTemplates(ageIdx, FeHIdx)
		
		return massLow, massUp, spectra, lum

class powerlawIMFprior:
	"""Class used to represent a power law IMF prior.
		
	This class represent a double power law IMF parameterization. The
	parameterization on the IMF is transformed into weights. For a 
	single power law simply use alpha1 = alpha2.
	
	Args:
		massLow: numpy array with the low mass boundaries of the
			isochrone mass bins.
		massUp: numpy array with the upper mass boundaries of the 
			isochrone mass bins.
		alpha1: low mass slope of the broken powerlaw IMF.
		alpha2: high mass slope of the broken powerlaw IMF.
		norm: normalization of the IMF.
		Mbreak: mass that defines break between power laws defined
			by alpha1 and alpha2 (optional, Mbreak = 0.5 by default).

	"""
	
	def __init__(self, massLow, massUp, alpha1, alpha2, norm, Mbreak=None):
		self.massLow = massLow
		self.massUp = massUp
		self.nTemplates = len(massLow)
		self.weights = np.zeros(self.nTemplates)
		self.alpha1 = alpha1
		self.alpha2 = alpha2
		self.norm = norm
		
		# Check if Mbreak is specified, otherwise use Mbreak = 0.5 Msun
		if Mbreak == None:
			self.Mbreak = 0.5
		else:
			self.Mbreak = Mbreak
		
		self.getWeightsDoublePL()
					
	def getWeightsDoublePL(self):
		"""Determine prior weights for double power law.
		
		This function determines the prior weights for a 
		double power law IMF parameterization.
		
		"""
		
		# Correction term normalization high mass end IMF to
		# ensure a continuous IMF.
		C2T = (self.Mbreak**(self.alpha2-self.alpha1))
		
		# Determine number of stars for each of the stellar templates by
		# integrating from the low mass end of the bin to the high mass
		# end of the bin.
		for template in range(self.nTemplates):
			# Low mass end
			if (self.massLow[template] < self.Mbreak and self.massUp[template] < self.Mbreak):
				self.weights[template] = ( (self.massLow[template])**(-self.alpha1+1) - (self.massUp[template])**(-self.alpha1+1) ) / (self.alpha1-1)
			# High mass end
			elif (self.massLow[template] > self.Mbreak and self.massUp[template] > self.Mbreak):
				self.weights[template] = C2T * ( (self.massLow[template])**(-self.alpha2+1) - (self.massUp[template])**(-self.alpha2+1) ) / (self.alpha2-1)
			# Transition region
			else:
				self.weights[template] = ( ((self.massLow[template])**(-self.alpha1+1) - (self.Mbreak)**(-self.alpha1+1) ) / (self.alpha1-1) ) + C2T * ( ((self.Mbreak)**(-self.alpha2+1) - (self.massUp[template])**(-self.alpha2+1) ) / (self.alpha2-1))
		
		self.weights *= self.norm
		
class SSPmodel:
	"""Class for creating SSP-model spectrum.
	
	This class may be used to create an SSP model spectrum
	with a double power law IMF.
		
	Args:
		templatesDir: directory with stellar templates.
		logAgeArr: numpy array with logAge-values of age-metallicity grid.
		FeHArr: numpy array with FeH-values of age-metallicity grid.
		age: required age of SSP (in Gyr).
		FeH: required [Fe/H] of SSP.
		alpha1: low-mass slope of the IMF.
		alpha2: high-mass slope of the IMF.
		normIMF: normalization of the IMF.
		Mbreak: break between two IMF slopes (optional, 0.5 by default).
		
	"""
	
	def __init__(self, templatesDir, logAgeArr, FeHArr, age, FeH, alpha1, alpha2, normIMF):
		# Read stellar templates
		stTemplates = stellarTemplates(templatesDir, logAgeArr, FeHArr)
		self.massLow, self.massUp, self.spectra, self.lum = stTemplates.getTemplates(age, FeH)
		
		# Determine weights from IMF
		self.alpha1, self.alpha2, self.normIMF = alpha1, alpha2, normIMF
		self.prior = powerlawIMFprior(self.massLow, self.massUp, alpha1=self.alpha1, alpha2=self.alpha2, norm=self.normIMF)
		
		# Calculate SSP-spectrum
		self.spectrum = np.dot(self.spectra, self.prior.weights)
		
	def addGaussianNoise(self, SNR):
		"""Add Gaussian noise to SSP model spectrum.
		
		Args:
			SNR: signal-to-noise ratio per bin.
			
		Returns:
			errorSpec: the error spectrum.
			
		"""
		avFlux = np.average(self.spectrum)
		counts = (self.spectrum / avFlux) * SNR**2
		self.spectrum = np.random.normal(counts, np.sqrt(counts)) * avFlux / SNR**2
		
		errorSpec = np.sqrt(counts) * avFlux / SNR**2
		
		return errorSpec
		
	def smoothSpectrum(self, wavelength, sigma):
		"""Smooth spectrum to given velocity dispersion.
		
		Args:
			sigma: velocity dispersion in km/s
		
		"""
		
		self.spectrum = specBasics.smoothSpectrum(wavelength, self.spectrum, sigma)
		
class respFunctions:
	"""Class for dealing with response functions.
	
	This class can be used to load the required response functions and
	to apply response functions of various elements to SSP spectra.
	
	Args:
		hdf5File: filename of hdf5-file with response functions
		abVariations: dictionary containing which response functions should be modelled.
		
	"""
	
	def __init__(self, nSSPs, abVariations, wavelength, goodIdx, hdf5File):
		# Check init-values
		self.nSSPs = nSSPs
		self.abVariations = abVariations
		self.wavelength = wavelength
		if all(value == False for value in list(self.abVariations.values())):
			self.applyResp = False
		else:
			self.applyResp = True
			if hdf5File==None:
				raise sys.exit("\nERROR: specify name of HDF5-file with response functions.\n")
			elif not os.path.isfile(hdf5File):
				raise sys.exit("\nERROR: HDF5-file \'" + hdf5File + "\' with response functions does not exist.\n")
			else:
				self.hdf5File = hdf5File
			
		# Load response functions from hdf5-file, if required
		if self.applyResp:
			self.loadRespFunctions(goodIdx)
			
	def loadRespFunctions(self, goodIdx):
		resHDF5File = h5py.File(self.hdf5File)
		if self.abVariations['Mg']:
			self.MgP = np.array(resHDF5File["MgP"])[:,:,goodIdx]
			self.MgM = np.array(resHDF5File["MgM"])[:,:,goodIdx]
		if self.abVariations['Ca']:	
			self.CaP = np.array(resHDF5File["CaP"])[:,:,goodIdx]
			self.CaM = np.array(resHDF5File["CaM"])[:,:,goodIdx]
		if self.abVariations['Si']:
			self.SiP = np.array(resHDF5File["SiP"])[:,:,goodIdx]
			self.SiM = np.array(resHDF5File["SiM"])[:,:,goodIdx]
		if self.abVariations['Ti']:
			self.TiP = np.array(resHDF5File["TiP"])[:,:,goodIdx]
			self.TiM = np.array(resHDF5File["TiM"])[:,:,goodIdx]
		if self.abVariations['Na']:
			self.NaP = np.array(resHDF5File["NaP"])[:,:,goodIdx]
			self.NaM = np.array(resHDF5File["NaM"])[:,:,goodIdx]
		resHDF5File.close()
			
	def applyRespFunctions(self, templatesList, ageIndices, FeHIndices, respDex):
		if self.applyResp:
			for tIdx in range(self.nSSPs):
				if self.abVariations['Mg']:
					if respDex['MgDex'] >= 0.0:
						MgRF = ((1.0 + respDex['MgDex']*(self.MgP[ageIndices[tIdx],FeHIndices[tIdx],:]-1.0)/0.3).reshape((len(self.wavelength),1)))
					else:
						MgRF = ((1.0 - respDex['MgDex']*(self.MgM[ageIndices[tIdx],FeHIndices[tIdx],:]-1.0)/0.3).reshape((len(self.wavelength),1)))
					templatesList[tIdx] *= MgRF
				if self.abVariations['Ca']:
					if respDex['CaDex'] >= 0.0:
						CaRF = ((1.0 + respDex['CaDex']*(self.CaP[ageIndices[tIdx],FeHIndices[tIdx],:]-1.0)/0.3).reshape((len(self.wavelength),1)))
					else:
						CaRF = ((1.0 - respDex['CaDex']*(self.CaM[ageIndices[tIdx],FeHIndices[tIdx],:]-1.0)/0.3).reshape((len(self.wavelength),1)))
					templatesList[tIdx] *= CaRF
				if self.abVariations['Si']:
					if respDex['SiDex'] >= 0.0:
						SiRF = ((1.0 + respDex['SiDex']*(self.SiP[ageIndices[tIdx],FeHIndices[tIdx],:]-1.0)/0.3).reshape((len(self.wavelength),1)))
					else:
						SiRF = ((1.0 - respDex['SiDex']*(self.SiM[ageIndices[tIdx],FeHIndices[tIdx],:]-1.0)/0.3).reshape((len(self.wavelength),1)))
					templatesList[tIdx] *= SiRF
				if self.abVariations['Ti']:
					if respDex['TiDex'] >= 0.0:
						TiRF = ((1.0 + respDex['TiDex']*(self.TiP[ageIndices[tIdx],FeHIndices[tIdx],:]-1.0)/0.3).reshape((len(self.wavelength),1)))
					else:
						TiRF = ((1.0 - respDex['TiDex']*(self.TiM[ageIndices[tIdx],FeHIndices[tIdx],:]-1.0)/0.3).reshape((len(self.wavelength),1)))
					templatesList[tIdx] *= TiRF
				if self.abVariations['Na']:
					if respDex['NaDex'] >= 0.0:
						NaRF = ((1.0 + respDex['NaDex']*(self.NaP[ageIndices[tIdx],FeHIndices[tIdx],:]-1.0)/0.3).reshape((len(self.wavelength),1)))
					else:
						NaRF = ((1.0 - respDex['NaDex']*(self.NaM[ageIndices[tIdx],FeHIndices[tIdx],:]-1.0)/0.3).reshape((len(self.wavelength),1)))
					templatesList[tIdx] *= NaRF
