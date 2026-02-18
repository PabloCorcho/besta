from abc import ABC, abstractmethod


class ModelGridGenerator(ABC):
    """
    Abstract interface for producing a ModelGrid from inputs.

    Implementations may synthesise spectra, SEDs, photometry, or any
    observable set, and attach associated continuous targets such as
    SFR, sSFR, mass, metallicity, age, or redshift.

    Subclasses should implement the generate method.
    """

    @abstractmethod
    def generate(self, **kwargs) -> "ModelGrid":
        """
        Build and return a ModelGrid.

        Parameters
        ----------
        **kwargs
            Implementation-specific keyword arguments such as parameter
            ranges, library choices, and resolution settings.

        Returns
        -------
        grid : ModelGrid
            A populated model grid instance.
        """
        raise NotImplementedError
