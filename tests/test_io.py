import os
import numpy as np
import pytest
from astropy.table import Table

from cosmosis import DataBlock

from besta import io
from besta.pipeline import MainPipeline


def test_reader_solution_to_datablock_fills_missing():
    reader = io.Reader.__new__(io.Reader)
    reader.ini_values_fixed = {("parameters", "foo"): 1.23}
    reader.ini_values_free = {("parameters", "bar"): (1, 10)}
    db = reader.solution_to_datablock({"parameters--bar": 5.0})
    assert db["parameters", "foo"] == 1.23
    assert db["parameters", "bar"] == 5.0

def test_parse_value_preserves_top_level_grouping():
    parsed = io._parse_value("(1, 2, 3), 2")
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    np.testing.assert_array_equal(parsed[0], np.array([1, 2, 3]))
    assert parsed[1] == 2


def test_parse_value_supports_bracketed_grouping():
    parsed = io._parse_value("[1, 2, 3], 2")
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    np.testing.assert_array_equal(parsed[0], np.array([1, 2, 3]))
    assert parsed[1] == 2


def test_parse_value_mixed_group_and_string():
    parsed = io._parse_value("foo, [1, 2, 3], 2")
    assert isinstance(parsed, list)
    assert parsed[0] == "foo"
    np.testing.assert_array_equal(parsed[1], np.array([1, 2, 3]))
    assert parsed[2] == 2


def test_string_to_func_args_parses_args_and_kwargs():
    args, kwargs = io.string_to_func_args("1, 2, x='something'")
    assert args == [1, 2]
    assert kwargs == {"x": "something"}


def test_string_to_func_args_supports_grouped_values():
    args, kwargs = io.string_to_func_args("(1, 2, 3), x=[4, 5, 6]")
    assert len(args) == 1
    np.testing.assert_array_equal(args[0], np.array([1, 2, 3]))
    np.testing.assert_array_equal(kwargs["x"], np.array([4, 5, 6]))


def test_string_to_func_args_tolerates_spaces_around_equals():
    args, kwargs = io.string_to_func_args("1, x = 'something', y = 2")
    assert args == [1]
    assert kwargs == {"x": "something", "y": 2}


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

if __name__ == "__main__":
    from besta.logging import setup_logging
    setup_logging()
    pytest.main([__file__])

@pytest.mark.parametrize("text", [
    "(0.3, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999), min_last_interval=0.0001, use_transforms=T",
    "(1, 2), [3.5, 4], name='two words'",
    "PADOVA00, imf=KROUPA_UNIVERSAL",
    "(0.5), 7",
])
def test_ini_value_to_string_round_trip(text):
    parsed = io._parse_value(text)
    again = io._parse_value(io.ini_value_to_string(parsed))
    assert type(again) is type(parsed)
    if isinstance(parsed, list):
        assert len(again) == len(parsed)
        for a, b in zip(again, parsed):
            if isinstance(b, np.ndarray):
                np.testing.assert_array_equal(a, b)
            else:
                assert a == b


def test_datablock_safe_options_for_ragged_values():
    # SFHArgs with keyword arguments parses to [array, str, str], which a
    # DataBlock cannot store (regression: besta-postprocess --make_best_fit)
    config = {"FullSpectralFit": {
        "sfhargs": io._parse_value("(0.3, 0.5, 0.999), min_last_interval=0.0001"),
        "wlrange": io._parse_value("3850 8900"),
        "redshift": 0.02,
        "sspmodelargs": io._parse_value("PADOVA00, imf=KROUPA_UNIVERSAL"),
    }}
    safe = io.datablock_safe_options(config)
    block = DataBlock.from_dict(safe)
    text = block["FullSpectralFit", "sfhargs"]
    assert isinstance(text, str)
    args, kwargs = io.string_to_func_args(text)
    np.testing.assert_allclose(np.ravel(args[0]), [0.3, 0.5, 0.999])
    assert float(kwargs["min_last_interval"]) == 1e-4
    np.testing.assert_array_equal(block["FullSpectralFit", "wlrange"], [3850, 8900])
    assert block["FullSpectralFit", "redshift"] == 0.02
    # The input configuration is not modified
    assert isinstance(config["FullSpectralFit"]["sfhargs"], list)
