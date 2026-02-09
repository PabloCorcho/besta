import os
import numpy as np
import pytest
from astropy.table import Table

from cosmosis import DataBlock

from besta import spectrum, postprocess, io
from besta.pipeline import MainPipeline


def test_mask_telluric_regions():
    wl = np.linspace(6000, 8000, 10)
    flux = np.ones_like(wl)
    err = np.ones_like(wl) * 0.1
    weights = np.ones_like(wl)
    new_w, mask = spectrum.mask_telluric_regions(
        wl, flux, err, weights, return_mask=True, pad=0.0
    )
    # Ensure some mask applied within known band (~6860-6950)
    assert mask.any()
    assert np.all(new_w[mask] == 0.0)


def test_mask_strong_emission_lines():
    wl = np.linspace(4990, 5010, 50)
    flux = np.zeros_like(wl)
    err = np.ones_like(wl) * 0.1
    weights = np.ones_like(wl)
    # Inject a "line" at center
    flux[25] = 10.0
    new_w, mask = spectrum.mask_strong_emission_lines(
        wl, flux, err, weights, redshift=0.0, return_mask=True, pad=0.0
    )
    assert mask.any()
    assert np.all(new_w[mask] == 0.0)


def test_legendre_decorator_applies_coeffs():
    wl = np.linspace(4000, 5000, 5)
    leg_pol = spectrum.get_legendre_polynomial_array(wl, order=2, bounds=None, clip_first_zero=False)

    class Dummy:
        def __init__(self):
            self.config = {"legendre_pol": leg_pol}

        @spectrum.legendre_decorator
        def make_observable(self, block, parse=False):
            return np.ones_like(wl)

    blk = DataBlock()
    blk["legendre", "legendre_1"] = 1.0
    blk["legendre", "legendre_2"] = 0.5
    out = Dummy().make_observable(blk)
    # Output should not be all ones after applying polynomials
    assert not np.allclose(out, 1.0)


def test_weighted_1d_cmf_raises_on_zero_weights():
    with pytest.raises(ValueError):
        postprocess.weighted_1d_cmf(np.array([1.0, 2.0]), np.array([0.0, 0.0]))


def test_compute_pdf_handles_minimal_table(tmp_path):
    tbl = Table()
    tbl["parameters--x"] = [0.1, 0.2, 0.3]
    tbl["post"] = [0.0, 1.0, 2.0]
    outfile = tmp_path / "pdf.fits"
    hdul = postprocess.compute_pdf_from_results(tbl, output_filename=str(outfile))
    assert "PERCENTILES" in hdul
    assert outfile.exists()


def test_reader_selection_helpers():
    reader = io.Reader.__new__(io.Reader)
    reader.results_table = Table()
    reader.results_table["parameters--a"] = [1.0, 2.0, 3.0]
    reader.results_table["post"] = [0.0, 0.5, 1.0]
    reader.ini_values = {"parameters": {"a": 0.0}}
    top = reader.get_top_frac_solutions(frac=10)
    assert len(top) == 1
    pct = reader.get_pct_solutions(pct=90)
    assert len(pct) > 0


def test_reader_solution_to_datablock_fills_missing():
    reader = io.Reader.__new__(io.Reader)
    reader.ini_values_fixed = {("parameters", "foo"): 1.23}
    reader.ini_values_free = {("parameters", "bar"): (1, 10)}
    db = reader.solution_to_datablock({"parameters--bar": 5.0})
    assert db["parameters", "foo"] == 1.23
    assert db["parameters", "bar"] == 5.0


def test_pipeline_execute_all_stops_on_failure(tmp_path, monkeypatch):
    cfg = {
        "output": {"filename": str(tmp_path / "out")},
        "pipeline": {"modules": "Dummy", "values": str(tmp_path / "vals.ini")},
        "Dummy": {"file": __file__},
    }

    class DummyPipeline(MainPipeline):
        def run_command(self, command):
            return 1

    pipe = DummyPipeline([cfg])
    # Prevent writing actual files
    monkeypatch.setattr(io, "make_ini_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(io, "make_values_file", lambda *args, **kwargs: None)
    assert pipe.execute_all() == 1
