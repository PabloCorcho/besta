import cosmosis
from cosmosis.datablock import option_section

class SFHModule(cosmosis.ClassModule):

    def load_data(self):
        
    def initialise_ssp(self):
        pass

class SFHMassBins(cosmosis.ClassModule):
    def __init__(self, options):
        self.setup(options)

    def setup(self, options):
        fileName = options.get_string("sfh_mass_bins", "inputSpectrum")
        ssp_name = options.get_string("SSPModel")
        if options.has_value()