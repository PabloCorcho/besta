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