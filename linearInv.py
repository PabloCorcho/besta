import numpy as np
import scipy.optimize
import scipy.linalg

class invertCSP:
	"""This class deals with the linear inversion of the hierarchical
		Bayesian framework.
		
		Args:
			wavelength: numpy array representing wavelength array of the data.
			flux: numpy array with the flux of the stellar population
				that will be modelled.
			SpecTemplates: matrix with stellar templates in columns, shape of
				the array should be consistent with flux.
			covarianceMatrix: numpy array with diagonal elements covariance matrix.
			w0: prior on the weights
			regScheme: choice of regularization scheme.

	"""
	
	def __init__(self, wavelength, flux, cov, specTemplates, w0, regScheme=1):
		# Define basic data members of class
		self.wavelength = wavelength
		self.flux = flux
		self.cov = cov
		self.error = np.sqrt(self.cov)
		self.specTemplates = np.copy(specTemplates)
		self.nTemplates = specTemplates.shape[1]
		self.nBins = specTemplates.shape[0]
		self.w0 = np.copy(w0)
		self.regScheme = regScheme
		
		# Calculate inverse of (diagonal) covariance matrix
		self.covInv = 1.0 / self.cov		
		
		# Create transposed matrix with stellar templates. Divide each element
		# by the diagonal elements of the covariance matrix.
		self.specTempT = self.__createSpecTempT()
				
		# Create matrix B: B = S^T x Cd^{-1} x S (see Dries et al. 2016).
		self.B = self.specTempT.dot(self.specTemplates)
		
		# Initialize matrix C with regularization scheme.
		self.__createMatC()
		
		# Define array with trial values of log lambda for finding
		# approximate minimum of evidence(lambda).
		self.nStartValues = 15
		self.logLamStart = np.linspace(-5,25,self.nStartValues)
		self.minEvidence = np.zeros(self.nStartValues)
		self.startIdx = 0
		self.startValue = 1.0
		
		# Calculate constant terms expression Bayesian evidence
		self.detCov = np.sum(np.log(self.cov))
		self.BE = -(self.specTemplates.shape[0] / 2) * np.log(2*np.pi)
		
		# Derive most probable regularization parameter
		self.lam = self.__bestLambda()
											
		# Define matrix A and calculate most probable distribution of weights.
		self.A = self.B + self.lam*self.C
		self.wMP = self.__MP(self.A, self.lam)
		
		# Define a number of class data members to store the different elements
		# that go into the calculation of the log evidence.
		self.chiSq = 0
		self.penaltyTerm = 0
		self.logDetA = 0
		self.p1 = 0
		self.p2 = 0
		self.p3 = 0

	# Private functions
	def __createSpecTempT(self):
		"""Create CD^{-1}*ST (see Dries et al 2016).
		
		This function calculates the transpose of the stellar
		templates matrix and multiplies the elements with the
		inverse covariance matrix.
		
		"""
		specTempT = np.copy(self.specTemplates)
		specTempT = np.transpose(specTempT)
		specTempT = specTempT * self.covInv
				
		return specTempT		
	
	def __createMatC(self):
		"""Initialize matrix C

		"""
		if self.regScheme != 1 and self.regScheme != 2:
			print("\nWARNING: regularization scheme not recognized, using identity matrix by default.\n")
		
		self.C = np.identity(self.nTemplates)
		if self.regScheme == 2:
			for id1 in range(self.nTemplates):
				self.C[id1,id1] = 1.0 / self.w0[id1]**2
				
	def __updateMatC(self):
		"""Update matrix C with new prior
		
		"""
		if self.regScheme == 2:
			for id1 in range(self.nTemplates):
				self.C[id1,id1] = 1.0 / self.w0[id1]**2
			
	def __MP(self, A, lam):
		""" Calculate most probable distribution of weights.
		
		Args:
			A: matrix A = B+lambda*C (see Dries et al. 2016)
			lam: regularization parameter.
		Returns:
			wMP: most probable distribution of weights.
		
		"""
	
		try:
			wMP = np.linalg.solve(self.B + lam*self.C, np.dot(self.specTempT, self.flux) + lam*np.dot(self.C, self.w0))
		except np.linalg.linalg.LinAlgError:
			# If somehow determination of solution is not possible, use vector
			# with zeros for weights which will give very low likelihood.
			wMP = np.zeros(self.nTemplates)
			
		return wMP
	
	def __bestLambda(self):
		"""Find most probable regularization parameter.
		
		"""
		
		# Determine starting value for brent-method (to avoid local minimum).
		self.startValue = self.__findStartValue()
			
		# Check if there exists a minimum within the range of self.lamStart. 
		# Otherwise, use fmin because we cannot provide an interval. 
		if (self.startIdx != 0 and self.startIdx != self.nStartValues-1):
			s = scipy.optimize.brent(self.__minBayesianEvidence, brack=(self.logLamStart[self.startIdx-1], self.logLamStart[self.startIdx], self.logLamStart[self.startIdx+1]))
		else:
			s = scipy.optimize.fmin(self.__minBayesianEvidence, self.startValue, disp=False)[0]
		
		return 10**s
		
	def __findStartValue(self):
		"""Find starting value for optimization lambda.
		
		To prevent the lambda-optimizer from ending in a local minimum,
		calculate evidence for a wide-range of log-lambda values so 
		that approximate value of minimum can be determined.
		
		"""
		self.minEvidence = 1e120*np.ones(self.nStartValues)
		for idx1 in range(self.nStartValues):
			self.minEvidence[self.nStartValues-1-idx1] = self.__minBayesianEvidence(self.logLamStart[self.nStartValues-1-idx1])
			# If minEvidence > 1e100 a negative value has been found for wMP which
			# implies that lower values of lambda do not need to be considered.
			if self.minEvidence[self.nStartValues-1-idx1] > 1e100:
				break
		self.startIdx = np.argmin(self.minEvidence)
				
		return self.logLamStart[self.startIdx]
		
	def __minBayesianEvidence(self, logLam):
		"""Calculate negative value Bayesian evidence.
		
		This function calculates value of minus the Bayesian
		evidence. This function is specifically meant for the
		optimizer of log-lambda
		
		"""
		lam = 10**logLam

		# Calculate wMP for given regularization parameter.
		A = self.B + lam*self.C
		wMP = self.__MP(A,lam)
		
		# Calculate Bayesian evidence. If any of the weights is
		# below zero, return very high number. This is the 
		# implementation of the prior on lambda that prevents
		# negative weights.
		if (wMP >= 0).all():
			# Calculate different terms Bayesian evidence.
			chiSq = self.__getChiSq(wMP)
			penaltyTerm = 0.5 * lam * np.dot(np.dot(np.transpose(wMP - self.w0), self.C), (wMP - self.w0))
			logDetA = np.linalg.slogdet(A)
			p1 = -chiSq - penaltyTerm - 0.5 * logDetA[1]
			p2 = self.BE - 0.5*self.detCov
			p3 = (self.nTemplates/2.0) * (np.log(lam)) + 0.5*np.linalg.slogdet(self.C)[1]
		
			# Calculate BE with flat prior on lambda in logspace
			BE = p1 + p2 + p3 - np.log(lam)
			
			# Return minus the evidence.
			return -BE
		else:
			return 1e120
			
	def __getChiSq(self, wMP):
		"""Calculate chi-squared value fit.
		
		Args:
			wMP: most probable weights
		Returns:
			chiSq: chi-squared value fit.
		
		"""
		
		# Calculate reconstructed spectrum
		recSp = np.dot(self.specTemplates, wMP)
		
		# Determine residual and transpose
		res = (recSp-self.flux).reshape(self.nBins,1)
		resT = np.transpose(res)
		
		# Determine chi-squared value.
		chiSq = 0.5*np.dot((resT * self.covInv), res)[0][0]
				
		return chiSq
	
	
	# Public functions
	#
	def BayesianEvidence(self):
		"""Calculate Bayesian evidence.
		
		"""
		self.chiSq = self.__getChiSq(self.wMP)
		self.penaltyTerm = 0.5 * self.lam * np.dot(np.dot(np.transpose(self.wMP - self.w0), self.C), (self.wMP - self.w0))
		self.logDetA = np.linalg.slogdet(self.A)
		
		self.p1 = -self.chiSq - self.penaltyTerm - 0.5 * self.logDetA[1]
		self.p2 = self.BE - 0.5*self.detCov
		self.p3 = (self.nTemplates/2.) * (np.log(self.lam)) + 0.5*np.linalg.slogdet(self.C)[1]
				
		BE = self.p1 + self.p2 + self.p3
		
		return BE
	
	def updateTemplatesCov(self, templates, cov):
		""" Update stellar templates and covariance matrices.
		
		Args:
			templates: new set of stellar templates.
			cov: (new) covariance matrix.
			
		"""
		
		# Update templates
		self.specTemplates = np.copy(templates)
		self.nTemplates = self.specTemplates.shape[1]
		
		# Update covariance
		self.cov = cov
		self.covInv = 1.0 / self.cov
		
		# Update specTempT, matrix B and determinant covariance matrix
		self.specTempT = self.__createSpecTempT()
		self.B = self.specTempT.dot(self.specTemplates)
		self.detCov = np.sum(np.log(self.cov))
		
	def updatePrior(self, w0):
		"""Update weights prior.
		
		Args:
			w0: new prior on the weights.
		
		"""
		
		# Update w0
		self.w0 = np.copy(w0)
		
		# Update matrix C
		self.__updateMatC()

		# Determine lambda, matrix A and most probable weights
		self.lam = self.__bestLambda()
		self.A = self.B + self.lam*self.C
		self.wMP = self.__MP(self.A, self.lam)
		