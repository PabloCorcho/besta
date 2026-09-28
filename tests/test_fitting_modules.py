import numpy as np
from astropy import units as u
import pytest

from cosmosis import DataBlock

from besta import kinematics, sfh
from besta.pipeline_modules.base_module import (
    SpectraFitModule,
    effective_lsf_sigma,
    rest_frame_instrumental_fwhm,
)
from besta.pipeline_modules.full_spectral_fit import FullSpectralFitModule
from besta.pipeline_modules.galaxy_spectra import GalaxySpectraModule
from besta.pipeline_modules.galaxy_photometry import GalaxyPhotometryModule
from besta.pipeline_modules.spectra_redshift_fit import SpectraRedshiftFitModule
import importlib


def make_dummy_spectrum(tmp_path):
    wl = np.linspace(4000, 5000, 50)
    flux = np.ones_like(wl)
    err = np.ones_like(wl) * 0.1
    fname = tmp_path / "spec.dat"
    np.savetxt(fname, np.vstack([wl, flux, err]).T)
    return str(fname)


def test_prepare_observed_spectra_weights_guard(tmp_path):
    spec = make_dummy_spectrum(tmp_path)

    class Dummy(SpectraFitModule):
        name = "Dummy"

        def __init__(self, options):
            super().__init__(options)
            options = self.parse_options(options)
            self.prepare_observed_spectra(options)

        def make_observable(self, *args, **kwargs):
            pass

        def execute(self, *args, **kwargs):
            pass

        def plot_solution(self, *args, **kwargs):
            pass

    # Zero weights should trigger ValueError
    opts = {
        "Dummy": {
            "inputSpectrum": spec,
            "mask": spec,  # reuse shape; will all be >0
            "wlUnits": "Angstrom",
            "fluxUnits": "1e-16 erg / (s cm2 Angstrom)",
            "wlRange": [4010, 4990],
        }
    }
    # overwrite mask file with zeros
    np.savetxt(spec, np.vstack([np.linspace(4000, 5000, 50), np.ones(50), np.ones(50)]).T)
    np.savetxt(tmp_path / "mask.dat", np.zeros(50))
    opts["Dummy"]["mask"] = str(tmp_path / "mask.dat")
    with pytest.raises(ValueError):
        Dummy(opts)


def test_emission_lines_masked_at_rest_frame_position(tmp_path):
    """Emission-line masks must land on the rest-frame line centre.

    ``prepare_observed_spectra`` de-redshifts the wavelength array before
    masking, so the lines must not be shifted by (1 + z) again. Previously
    H-alpha at z = 0.1 was searched for at 6563 * 1.1 = 7219 AA on the
    rest-frame axis (here outside the range, so nothing was masked).
    """
    z = 0.1
    rest_wl = np.arange(6200.0, 6900.0, 1.0)
    obs_wl = rest_wl * (1 + z)
    flux = np.ones_like(obs_wl)
    # Strong H-alpha emission at the observed position
    flux += 5.0 * np.exp(-0.5 * ((obs_wl - 6562.80 * (1 + z)) / 3.0) ** 2)
    err = np.full_like(obs_wl, 0.05)
    spec = tmp_path / "spec_emission.dat"
    np.savetxt(spec, np.vstack([obs_wl, flux, err]).T)

    class Dummy(SpectraFitModule):
        name = "Dummy"

        def __init__(self, options):
            super().__init__(options)
            options = self.parse_options(options)
            self.prepare_observed_spectra(options)

        def make_observable(self, *args, **kwargs):
            pass

        def execute(self, *args, **kwargs):
            pass

        def plot_solution(self, *args, **kwargs):
            pass

    mod = Dummy({
        "Dummy": {
            "inputSpectrum": str(spec),
            "redshift": z,
            "wlRange": [6250.0, 6850.0],  # rest frame
            "mask_emission_lines": True,
        }
    })

    wl = mod.config["wavelength"].to_value("Angstrom")
    mask = mod.config["emission_lines_mask"]
    names = [line.name for line in mod.config["emission_lines_used"]]

    assert "Ha" in names
    assert mask[np.argmin(np.abs(wl - 6562.80))]
    # Only the H-alpha/[NII] complex is masked, nothing elsewhere
    assert np.all(np.abs(wl[mask] - 6562.80) < 40.0)
    assert np.all(mod.config["weights"][mask] == 0.0)


@pytest.mark.parametrize("with_features", [True, False])
def test_redshift_sweep_weights_apply_features_once(with_features):
    """Feature weights must enter the sweep and the likelihood exactly once.

    Previously ``SpectraRedshiftFit`` recomputed the feature weights on
    already feature-weighted weights and multiplied them in again (w**2).
    """
    mask = np.array([1.0, 1.0, 0.5, 0.0, 1.0])
    feat = np.array([0.0, 1.0, 0.8, 1.0, 0.2])
    var = np.full(5, 2.0)

    mod = SpectraRedshiftFitModule.__new__(SpectraRedshiftFitModule)
    mod.config = {
        "flux": np.ones(5),
        "mask_weights": mask.copy(),
        "norm_obs_var": var,
    }
    if with_features:
        # As left by prepare_observed_spectra(use_features=T)
        mod.config["feature_weights"] = feat.copy()
        mod.config["weights"] = mask * feat
        expected_sweep = feat * mask / var
    else:
        mod.config["weights"] = mask.copy()
        expected_sweep = mask / var
    weights_before = mod.config["weights"].copy()

    mod._setup_sweep_weights()

    np.testing.assert_allclose(mod.config["sweep_weights"], expected_sweep)
    # Likelihood weights are not modified (features applied once, upstream)
    np.testing.assert_allclose(mod.config["weights"], weights_before)
    # "Original" weights are the masking-only weights
    np.testing.assert_allclose(mod.config["weights_orig"], mask)
    np.testing.assert_array_equal(
        mod.config["good_idx"], np.flatnonzero(expected_sweep > 0))


def _make_features_dummy_module(tmp_path, noise):
    """Dummy spectral module on a flat spectrum with one absorption line."""
    wl = np.arange(4000.0, 5000.0, 1.0)
    flux = 1.0 - 0.3 * np.exp(-0.5 * ((wl - 4500.0) / 5.0) ** 2)
    err = np.full_like(wl, 0.01)
    if noise:
        flux = flux + np.random.default_rng(1).normal(scale=err)
    spec = tmp_path / "spec_features.dat"
    np.savetxt(spec, np.vstack([wl, flux, err]).T)

    class Dummy(SpectraFitModule):
        name = "Dummy"

        def __init__(self, options):
            super().__init__(options)
            options = self.parse_options(options)
            self.prepare_observed_spectra(options)

        def make_observable(self, *args, **kwargs):
            pass

        def execute(self, *args, **kwargs):
            pass

        def plot_solution(self, *args, **kwargs):
            pass

    mod = Dummy({"Dummy": {"inputSpectrum": str(spec), "use_features": True}})
    return mod, wl


def test_prepare_observed_spectra_keeps_mask_weights_with_features(tmp_path):
    """``mask_weights`` holds the pre-feature weights; features applied once."""
    mod, wl = _make_features_dummy_module(tmp_path, noise=True)

    assert np.all(mod.config["mask_weights"] == 1.0)
    assert np.all(np.isfinite(mod.config["feature_weights"]))
    np.testing.assert_allclose(
        mod.config["weights"],
        mod.config["mask_weights"] * mod.config["feature_weights"])
    # The absorption feature gets a non-zero feature weight
    assert mod.config["feature_weights"][np.argmin(np.abs(wl - 4500.0))] > 0


def test_feature_weights_fall_back_to_ones_without_continuum_scatter(tmp_path):
    """Noiseless input gives zero continuum scatter: no NaN feature weights."""
    mod, _ = _make_features_dummy_module(tmp_path, noise=False)

    np.testing.assert_array_equal(mod.config["feature_weights"], 1.0)
    np.testing.assert_array_equal(mod.config["weights"], mod.config["mask_weights"])


class _FakeQuantity:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)

    def to_value(self, *args, **kwargs):
        return self.value


class _FakeSFH:
    sect_name = "stars.sfh"
    use_mass_normalization = False
    today = 13.0

    def __init__(self, n_model):
        n = n_model

        class _Model:
            @staticmethod
            def compute_SED(*args, **kwargs):
                return _FakeQuantity(np.ones(n))

            @staticmethod
            def stellar_mass_formed(*args, **kwargs):
                return _FakeQuantity(1.0)

        self.model = _Model()

    def parse_datablock(self, block):
        return 1, 0.0


class _FakeGalaxy:
    def __init__(self, n_model):
        self.n_model = n_model

    def update_parameters(self, *args, **kwargs):
        pass

    def emission_spectrum(self, *args, **kwargs):
        return _FakeQuantity(np.ones(self.n_model))


@pytest.mark.parametrize("module_class", [FullSpectralFitModule, GalaxySpectraModule])
@pytest.mark.parametrize("extra_pixels", [16, 0])
@pytest.mark.parametrize("los_sigma", [50.0, 300.0, 600.0])
def test_likelihood_pixel_set_independent_of_kinematics(
    module_class, extra_pixels, los_sigma
):
    """The likelihood pixel set must not change with the LOSVD parameters.

    velscale = 50 km/s and a 16-pixel (800 km/s) buffer: sigma = 600 km/s gives
    a kernel half-width of ~64 pixels, larger than the buffer. Previously the
    edge pixels were then dropped from the likelihood (and extra_pixels = 0
    returned an empty model).
    """
    n_obs = 200
    n_model = n_obs + 2 * extra_pixels
    mod = module_class.__new__(module_class)
    mod.config = {
        "sfh_model": _FakeSFH(n_model),
        "ssp_model": None,
        "galaxy": _FakeGalaxy(n_model),
        "dl_sq": 1.0,
        "extinction_law": None,
        "extra_pixels": extra_pixels,
        "velscale": 50.0,
        "flux": np.ones(n_obs),
        "weights": np.ones(n_obs),
    }
    mod._losvd_kernel = kinematics.GaussianPixelKernel(
        velocity_scale=50.0, sigma_truncation=5.0)

    block = DataBlock()
    block["kinematics", "los_vel"] = 100.0
    block["kinematics", "los_sigma"] = los_sigma

    flux_model, weights = mod.make_observable(block)

    assert flux_model.shape == (n_obs,)
    np.testing.assert_array_equal(weights, mod.config["weights"])
    # Constant SED stays constant after convolution (edge padding, no flux loss)
    np.testing.assert_allclose(flux_model, 1.0)


def test_full_spectral_fit_make_observable(tmp_path):
    spec = make_dummy_spectrum(tmp_path)
    block = DataBlock()
    all_params = {
        "dust.attenuation": {"a_v": 0.0},
        "kinematics":  {
        "los_vel": 0.0,
        "los_sigma": 100.0,
        "los_h3": 0.0,
        "los_h4": 0.0,
        },
        "stars.sfh": {
        "logtau": 0.5,
        "alpha_powerlaw": 1.0,
        "ism_metallicity_today": 0.02,
        }
    }
    for sect, params in all_params.items():
        for k, v in params.items():
            block[sect, k] = v

    opts = {
        "FullSpectralFit": {
            "inputSpectrum": spec,
            "SSPModel": "PopStar",
            "SSPModelArgs": "cha",
            "SSPDir": "None",
            "wlUnits": "Angstrom",
            "fluxUnits": "1e-16 erg / (s cm2 Angstrom)",
            "wlRange": [4010, 4990],
            "velscale": 200.0,
            "ExtinctionLaw": "ccm89",
            "SFHModel": "ExponentialSFH",
            "save_ssfr_over_tau": [0.1, 1.0],
        }
    }
    mod = FullSpectralFitModule(opts)
    flux_model, weights = mod.make_observable(block, parse=True)
    assert flux_model.shape == mod.config["flux"].shape
    assert weights.shape == mod.config["flux"].shape
    np.testing.assert_allclose(mod.config["ssfr_tau"], [0.1, 1.0])
    assert np.isfinite(block["extra", "ssfr_over_tau_0.1000"])
    assert np.isfinite(block["extra", "ssfr_over_tau_1.0000"])


def test_full_spectral_fit_fixed_time_uses_absolute_mass(tmp_path):
    spec = make_dummy_spectrum(tmp_path)
    block = DataBlock()
    parameters = {
        "dust.attenuation": {"a_v": 0.0},
        "kinematics": {
            "los_vel": 0.0,
            "los_sigma": 100.0,
            "los_h3": 0.0,
            "los_h4": 0.0,
        },
        "stars.sfh": {
            "alpha_powerlaw": 1.0,
            "ism_metallicity_today": 0.02,
            "logsfr_at_5.000": 0.0,
            "logsfr_at_2.000": 0.0,
            "logsfr_at_1.000": 0.0,
            "logsfr_at_0.500": 0.0,
        },
    }
    for section, values in parameters.items():
        for key, value in values.items():
            block[section, key] = value

    options = {
        "FullSpectralFit": {
            "inputSpectrum": spec,
            "SSPModel": "PopStar",
            "SSPModelArgs": "cha",
            "SSPDir": "None",
            "wlUnits": "Angstrom",
            "fluxUnits": "1e-16 erg / (s cm2 Angstrom)",
            "wlRange": [4010, 4990],
            "velscale": 200.0,
            "ExtinctionLaw": "ccm89",
            "SFHModel": "FixedTimeSFH",
            "SFHArgs": "[0.5, 1.0, 2.0, 5.0]",
        }
    }
    module = FullSpectralFitModule(options)
    flux_model, _ = module.make_observable(block, parse=True)
    formed_mass = module.config["sfh_model"].model.stellar_mass_formed(
        module.config["sfh_model"].today
    ).to_value("Msun")

    assert np.all(np.isfinite(flux_model))
    assert module.config["sfh_model"].use_mass_normalization is False
    assert block["extra", "stellar_mass"] == pytest.approx(
        np.log10(formed_mass)
    )


@pytest.mark.parametrize(
    "module_class, sets_stellar_mass",
    [
        (FullSpectralFitModule, True),
        (GalaxySpectraModule, True),
        (GalaxyPhotometryModule, True),
        (SpectraRedshiftFitModule, False),
    ],
)
def test_invalid_sample_is_rejected(module_class, sets_stellar_mass):
    """Invalid SFH samples must get a large *negative* log-likelihood.

    ``SFHBase.parse_datablock`` returns ``(0, -1e20)`` for invalid samples.
    Regression test for modules that stored ``-1e20 * penalty`` (= +1e40).
    """
    class InvalidSFH:
        @staticmethod
        def parse_datablock(block):
            return 0, -1e20  # invalid sample with a prior penalty

    mod = module_class.__new__(module_class)
    mod.config = {"sfh_model": InvalidSFH()}
    mod.like_name = "test_like"
    block = DataBlock()

    assert mod.execute(block) == 0
    assert block["likelihoods", mod.like_name] == -1e20
    if sets_stellar_mass:
        assert np.isnan(block["extra", "stellar_mass"])


@pytest.mark.parametrize(
    "module_class, is_photometry, saves_chi2",
    [
        (FullSpectralFitModule, False, True),
        (GalaxySpectraModule, False, True),
        (GalaxyPhotometryModule, True, False),
        (SpectraRedshiftFitModule, False, False),
    ],
)
@pytest.mark.parametrize("log_prior", [-3.5, 0.0, None])
def test_valid_sample_includes_sfh_log_prior(
    module_class, is_photometry, saves_chi2, log_prior
):
    """The SFH log-prior (e.g. smoothness prior) must be added to the likelihood.

    ``chi2`` diagnostics must stay likelihood-only (no prior contribution).
    """
    data_loglike = -10.0

    class ValidSFH:
        @staticmethod
        def parse_datablock(block):
            return 1, log_prior

    mod = module_class.__new__(module_class)
    mod.like_name = "test_like"
    mod.noise_model = None
    mod.log_like = lambda *args, **kwargs: data_loglike
    if is_photometry:
        mod.config = {
            "sfh_model": ValidSFH(),
            "photometry_flux": np.ones(3),
            "photometry_flux_var": np.ones(3),
            "photometry_lower_limit": None,
            "photometry_upper_limit": None,
        }
        mod.make_observable = lambda block: np.ones(3)
    else:
        mod.config = {
            "sfh_model": ValidSFH(),
            "flux": np.ones(5),
            "ivar": np.ones(5),
            "save_chi2": True,
        }
        mod.make_observable = lambda block: (np.full(5, 0.9), np.ones(5))
    block = DataBlock()

    assert mod.execute(block) == 0
    expected = data_loglike + (0.0 if log_prior is None else log_prior)
    assert block["likelihoods", mod.like_name] == pytest.approx(expected)
    if saves_chi2:
        assert block["extra", mod.like_name + "_chi2"] == pytest.approx(
            -2 * data_loglike
        )


def test_rest_frame_instrumental_fwhm_constant_lsf():
    """A constant observed FWHM shrinks by (1 + z) in the rest frame."""
    spec_rest_wl = np.linspace(4000.0, 5000.0, 101)
    lsf_obs = np.full_like(spec_rest_wl, 3.0)
    fwhm = rest_frame_instrumental_fwhm(
        [4200.0, 4800.0], spec_rest_wl, lsf_obs, redshift=1.0)
    np.testing.assert_allclose(fwhm, 1.5)


def test_rest_frame_instrumental_fwhm_preserves_resolving_power():
    """R = lambda / FWHM is frame-invariant; LSF is read at lambda * (1 + z).

    With FWHM_obs(lambda_obs) = lambda_obs / R, the rest-frame FWHM must be
    lambda_rest / R. Reading the LSF at lambda_rest (old bug) or skipping the
    (1 + z) factor both break this.
    """
    resolving_power, z = 1000.0, 0.5
    spec_rest_wl = np.linspace(4000.0, 5000.0, 1001)
    # instrumental FWHM at each pixel's *observed* wavelength
    lsf_obs = spec_rest_wl * (1 + z) / resolving_power
    query = np.array([4100.0, 4500.0, 4900.0])
    fwhm = rest_frame_instrumental_fwhm(query, spec_rest_wl, lsf_obs, z)
    np.testing.assert_allclose(fwhm, query / resolving_power, rtol=1e-6)


def test_effective_lsf_sigma_quadrature():
    sigma = effective_lsf_sigma([3.0, 4.0], [2.51, 0.0])
    expected = np.sqrt([(3.0 / 2.355) ** 2 - (2.51 / 2.355) ** 2,
                        (4.0 / 2.355) ** 2])
    np.testing.assert_allclose(sigma, expected)


def test_effective_lsf_sigma_clips_when_templates_are_broader():
    """Templates broader than the instrument: no broadening, no exception."""
    sigma = effective_lsf_sigma([2.28, 2.51, 3.0], [2.51, 2.51, 2.51],
                                wavelength=[3900.0, 5000.0, 6000.0])
    assert sigma[0] == 0.0
    assert sigma[1] == 0.0
    assert sigma[2] > 0.0
    assert np.all(np.isfinite(sigma))


def test_prepare_sfh_model_forwards_smoothness_prior_options():
    class Dummy(SpectraFitModule):
        name = "Dummy"

        def make_observable(self, *args, **kwargs):
            pass

        def execute(self, *args, **kwargs):
            pass

        def plot_solution(self, *args, **kwargs):
            pass

    mod = Dummy.__new__(Dummy)
    mod.alias = "Dummy"
    mod.config = {}
    options = mod.parse_options(
        {
            "Dummy": {
                "SFHModel": "FixedTimeSFH",
                "SFHArgs": "[0.5, 1.0, 2.0, 5.0]",
                "use_sfh_smoothness_prior": True,
                "sfh_smoothness_prior_type": "robust_time_curvature",
                "sfh_smoothness_sigma_dex": 0.7,
                "sfh_smoothness_dof": 5.0,
                "sfh_smoothness_relative_floor": 1e-6,
            }
        }
    )

    mod.prepare_sfh_model(options)

    prior = mod.config["sfh_model"].sfh_smoothness_prior
    assert isinstance(prior, sfh.SFHSmoothnessPrior)
    assert prior.sigma_dex == 0.7
    assert prior.dof == 5.0
    assert prior.relative_sfr_floor == 1e-6


def test_prepare_sfh_model_builds_fixed_mass_frac_2d():
    class Dummy(SpectraFitModule):
        name = "Dummy"

        def make_observable(self, *args, **kwargs):
            pass

        def execute(self, *args, **kwargs):
            pass

        def plot_solution(self, *args, **kwargs):
            pass

    mod = Dummy.__new__(Dummy)
    mod.alias = "Dummy"
    mod.config = {}
    options = mod.parse_options(
        {
            "Dummy": {
                "SFHModel": "FixedMassFracSFH2D",
                "SFHArgs": "[0.2, 0.5, 0.8]",
            }
        }
    )

    mod.prepare_sfh_model(options)

    model = mod.config["sfh_model"]
    assert isinstance(model, sfh.FixedMassFracSFH2D)
    assert model.model.name == "tabular_mass_frac_cem_2d"
    assert "sigma_log_metallicity" in model.free_params


if __name__ == "__main__":
    import sys
    import unittest

    unittest.main(argv=[sys.argv[0]])
