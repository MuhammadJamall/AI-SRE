from pathlib import Path

from src.common.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config, resolve


def test_default_config_contains_existing_dataset_paths():
    config = load_config()
    assert config["datasets"]["smd"]["raw_path"] == "data/raw/ServerMachineDataset"
    assert config["datasets"]["hdfs"]["raw_path"] == "data/raw/hdfs"
    assert DEFAULT_CONFIG.is_file()


def test_explicit_config_takes_precedence_over_environment(tmp_path, monkeypatch):
    env_config = tmp_path / "env.yaml"
    explicit_config = tmp_path / "explicit.yaml"
    env_config.write_text("source: environment\n", encoding="utf-8")
    explicit_config.write_text("source: explicit\n", encoding="utf-8")
    monkeypatch.setenv("AETHER_CONFIG", str(env_config))

    assert load_config(explicit_config) == {"source": "explicit"}


def test_environment_config_is_used_when_no_explicit_path(tmp_path, monkeypatch):
    config_path = tmp_path / "custom.yaml"
    config_path.write_text("source: environment\n", encoding="utf-8")
    monkeypatch.setenv("AETHER_CONFIG", str(config_path))

    assert load_config() == {"source": "environment"}


def test_resolve_uses_project_root_for_relative_paths():
    assert resolve("data/raw") == PROJECT_ROOT / "data/raw"


def test_resolve_preserves_absolute_paths(tmp_path):
    path = tmp_path / "raw-data"
    assert resolve(path) == path
