import pytest

from besta import io
from besta.pipeline import MainPipeline, BatchPipeline
import besta.pipeline as pipeline_module


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


def test_batch_pipeline_validate_input_lengths():
    configs = [[{"pipeline": {"modules": "Dummy"}, "Dummy": {}, "output": {"filename": "x"}}]]
    with pytest.raises(ValueError):
        BatchPipeline(
            pipeline_configuration_list=configs,
            n_cores_list=[1, 2],
        )


def test_batch_from_running_parameters_deepcopy_isolated_configs():
    base_config = {
        "pipeline": {"modules": "Dummy", "values": "values.ini"},
        "output": {"filename": "base"},
        "Dummy": {
            "file": "dummy.py",
            "nested": {"alpha": 1, "beta": 2},
        },
    }
    running_parameters = [
        {"Dummy": {"nested": {"alpha": 10}}},
        {"Dummy": {"nested": {"alpha": 20}}},
    ]

    batch = BatchPipeline.from_running_parameters(
        pipeline_configuration=base_config,
        running_parameters=running_parameters,
    )

    cfg_a = batch.all_pipelines_config[0][0]
    cfg_b = batch.all_pipelines_config[1][0]

    assert cfg_a["Dummy"]["nested"]["alpha"] == 10
    assert cfg_b["Dummy"]["nested"]["alpha"] == 20
    assert cfg_a["Dummy"]["nested"] is not cfg_b["Dummy"]["nested"]
    assert base_config["Dummy"]["nested"]["alpha"] == 1


def test_batch_from_running_parameters_rejects_invalid_inputs():
    base_config = {
        "pipeline": {"modules": "Dummy", "values": "values.ini"},
        "output": {"filename": "base"},
        "Dummy": {"file": "dummy.py"},
    }

    with pytest.raises(TypeError):
        BatchPipeline.from_running_parameters(
            pipeline_configuration=base_config,
            running_parameters={"Dummy": {"x": 1}},
        )

    with pytest.raises(TypeError):
        BatchPipeline.from_running_parameters(
            pipeline_configuration=base_config,
            running_parameters=[{"Dummy": {"x": 1}}, "bad-entry"],
        )


def test_batch_from_running_parameters_calls_keyword_only_constructor(monkeypatch):
    base_config = {
        "pipeline": {"modules": "Dummy", "values": "values.ini"},
        "output": {"filename": "base"},
        "Dummy": {"file": "dummy.py"},
    }

    captured = {}

    def fake_init(self, *, pipeline_configuration_list, **kwargs):
        captured["pipeline_configuration_list"] = pipeline_configuration_list
        captured["kwargs"] = kwargs

    monkeypatch.setattr(BatchPipeline, "__init__", fake_init)

    BatchPipeline.from_running_parameters(
        pipeline_configuration=base_config,
        running_parameters=[{"Dummy": {"x": 1}}],
        n_jobs_parallel=2,
    )

    assert "pipeline_configuration_list" in captured
    assert len(captured["pipeline_configuration_list"]) == 1
    assert captured["kwargs"]["n_jobs_parallel"] == 2


def test_batch_run_all_pipelines_sequential_order(monkeypatch):
    def fake_worker(job):
        return {"index": job["index"], "status": 100 + job["index"]}

    monkeypatch.setattr(pipeline_module, "_run_main_pipeline_job", fake_worker)

    configs = [
        [{"pipeline": {"modules": "Dummy"}, "Dummy": {}, "output": {"filename": "a"}}],
        [{"pipeline": {"modules": "Dummy"}, "Dummy": {}, "output": {"filename": "b"}}],
        [{"pipeline": {"modules": "Dummy"}, "Dummy": {}, "output": {"filename": "c"}}],
    ]

    batch = BatchPipeline(
        pipeline_configuration_list=configs,
        n_jobs_parallel=1,
    )

    assert batch.run_all_pipelines() == [100, 101, 102]

if __name__ == "__main__":
    from besta.logging import setup_logging
    setup_logging()
    pytest.main([__file__])

def test_pipeline_writes_physical_results(tmp_path):
    import numpy as np
    from test_postprocessing import _write_cosmosis_results

    path = str(tmp_path / "results.txt")
    rng = np.random.default_rng(0)
    _write_cosmosis_results(path, rng.uniform(size=(20, 3)),
                            rng.normal(size=(20, 3)))

    class _Reader:
        ini = io.Reader.read_ini_file_from_results(path)

    pipe = MainPipeline([{"pipeline": {"modules": "FullSpectralFit"},
                          "FullSpectralFit": {}}])
    output = pipe.write_physical_results(_Reader())
    assert output == str(tmp_path / "results_physical.txt")
    assert io.read_results_file(output).meta["besta_sfh_space"] == "physical"


def test_pipeline_physical_results_failure_is_logged(tmp_path, caplog):
    class _Reader:
        ini = {"output": {"filename": str(tmp_path / "missing.txt")}}

    pipe = MainPipeline([{"pipeline": {"modules": "Dummy"}, "Dummy": {}}])
    assert pipe.write_physical_results(_Reader()) is None


# --- SFH reconstruction in the pipeline managers ---------------------------

def _sfh_reader(tmp_path, name="results.txt"):
    import numpy as np
    from besta import postprocess
    from test_postprocessing import _write_cosmosis_results

    path = str(tmp_path / name)
    rng = np.random.default_rng(1)
    _write_cosmosis_results(path, rng.uniform(size=(30, 3)), rng.normal(size=(30, 3)))

    class _Reader:
        ini = io.Reader.read_ini_file_from_results(path)
        ini_values = postprocess._read_embedded_values(path)
        _results_table = io.read_results_file(path)
        results_file = path
        walkers = 5
    return _Reader()


def test_sfh_settings_validation():
    from besta.pipeline import normalize_sfh_reconstruction_settings as norm
    assert norm(None) is None and norm(False) is None
    settings = norm(True)
    assert settings["output"] is None and settings["options"] == {}
    assert norm({"samples": True, "options": {"n_bins": 10}})["options"] == {"n_bins": 10}
    with pytest.raises(ValueError):
        norm({"sample": True})                       # typo in a setting
    with pytest.raises(ValueError):
        norm({"options": {"nbins": 10}})             # typo in an option
    with pytest.raises(TypeError):
        norm("yes")


def test_pipeline_writes_sfh_reconstruction(tmp_path):
    from besta.pipeline import normalize_sfh_reconstruction_settings as norm
    from besta.postprocess import SFHReconstruction

    reader = _sfh_reader(tmp_path)
    pipe = MainPipeline([{"pipeline": {"modules": "FullSpectralFit"},
                          "FullSpectralFit": {}}])
    settings = norm({"plot": True, "samples": True,
                     "options": {"n_bins": 8, "burn_in": 2}})
    output = pipe.write_sfh_reconstruction(reader, settings, step=0)
    assert output == str(tmp_path / "results_sfh.fits")
    assert (tmp_path / "results_sfh.png").exists()
    rec = SFHReconstruction.from_fits(output)
    assert rec.lookback_edges.size == 9
    # 30 rows; burn-in of 2 steps x 5 walkers (from reader.walkers) -> 20 samples
    assert rec.n_samples == 20


def test_pipeline_sfh_explicit_output_per_step(tmp_path):
    from besta.pipeline import normalize_sfh_reconstruction_settings as norm
    reader = _sfh_reader(tmp_path)
    cfg = {"pipeline": {"modules": "FullSpectralFit"}, "FullSpectralFit": {}}
    pipe = MainPipeline([cfg, cfg])
    explicit = str(tmp_path / "my_sfh.fits")
    output = pipe.write_sfh_reconstruction(reader, norm({"output": explicit}), step=1)
    assert output == str(tmp_path / "my_sfh_step1.fits")
    single = MainPipeline([cfg])
    assert single.write_sfh_reconstruction(reader, norm({"output": explicit}), step=0) == explicit


def test_pipeline_sfh_reconstruction_skips_and_logs_failures(tmp_path):
    from besta.pipeline import normalize_sfh_reconstruction_settings as norm
    pipe = MainPipeline([{"pipeline": {"modules": "Dummy"}, "Dummy": {}}])

    class _NoSFH:
        ini = {"pipeline": {"modules": "Dummy"}, "Dummy": {},
               "output": {"filename": str(tmp_path / "x.txt")}}
    assert pipe.write_sfh_reconstruction(_NoSFH(), norm(True)) is None

    reader = _sfh_reader(tmp_path)
    reader._results_table = reader._results_table[:0]     # nothing to evaluate
    reader.ini_values = {"stars.sfh": {"alpha_powerlaw": [0, 1, 3]}}  # missing column
    assert pipe.write_sfh_reconstruction(reader, norm(True)) is None


def test_batch_pipeline_forwards_sfh_settings(monkeypatch):
    calls = []

    def fake_execute_all(self, plot_result=False, sfh_reconstruction=None):
        calls.append(sfh_reconstruction)
        return 0

    monkeypatch.setattr(MainPipeline, "execute_all", fake_execute_all)
    configs = [[{"pipeline": {"modules": "Dummy"}, "Dummy": {},
                 "output": {"filename": "x"}}]]
    batch = BatchPipeline(pipeline_configuration_list=configs)
    assert batch.run_single_pipeline(0, sfh_reconstruction={"samples": True}) == 0
    assert calls[-1]["samples"] is True and calls[-1]["options"] == {}
    with pytest.raises(ValueError):
        batch.run_single_pipeline(0, sfh_reconstruction={"sample": True})


def test_besta_run_sfh_flags():
    from besta.cli.besta_run import _sfh_settings, parser_setup
    parser = parser_setup()
    assert _sfh_settings(parser.parse_args(["cfg.ini"])) is None
    args = parser.parse_args(["cfg.ini", "--sfh", "--sfh-samples", "--sfh-n-bins", "12",
                              "--sfh-taus", "0.1,1", "--sfh-burn-in", "50"])
    settings = _sfh_settings(args)
    assert settings["samples"] is True and settings["plot"] is False
    assert settings["options"] == {"n_bins": 12, "taus": [0.1, 1.0], "burn_in": 50}
