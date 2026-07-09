"""CPU-only tests for titan_oellm.cluster_config.

These exercise the cluster-path/config resolution layer without importing torch
or torchtitan, so they run on a plain CPU machine:

    python3 tests/test_cluster_config.py

They are also discoverable by pytest (test_* functions).

The module is loaded directly from its source file to bypass
titan_oellm/__init__.py, which imports torchtitan (not needed here). We point the
loaded module's __file__ at a temporary directory so load_cluster_paths() resolves
user/cluster_paths.toml inside that temp dir instead of the real repo.
"""

import importlib.util
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLUSTER_CONFIG_SRC = REPO_ROOT / "titan_oellm" / "cluster_config.py"

MINIMAL_CLUSTER_PATHS = """
["cluster.local"]
output_dir = "/tmp/titan_test/out"
cache_base = "/tmp/titan_test/cache"

["tokenizer.neox.local"]
path = "/tmp/titan_test/tokenizer/neox"

["dataset.test_dataset.neox.local"]
train_prefix = "/tmp/titan_test/data/train"
train_chunks = "/tmp/titan_test/data/chunks"
validation_prefix = "/tmp/titan_test/data/validation"
dataloader = "MMapDataset"
min_doc_len = 128
"""


def _load_module(project_root: Path):
    """Load cluster_config.py with __file__ rooted at project_root/titan_oellm/."""
    spec = importlib.util.spec_from_file_location(
        "cluster_config_under_test", CLUSTER_CONFIG_SRC
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # load_cluster_paths() does Path(__file__).parent.parent / "user" / ...
    module.__file__ = str(project_root / "titan_oellm" / "cluster_config.py")
    return module


def _make_project(tmp: Path, with_config: bool = True):
    (tmp / "titan_oellm").mkdir(parents=True, exist_ok=True)
    if with_config:
        user_dir = tmp / "user"
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / "cluster_paths.toml").write_text(MINIMAL_CLUSTER_PATHS)
    return _load_module(tmp)


def test_tomllib_backend_available():
    """The module imports its TOML parser (stdlib tomllib or tomli)."""
    with tempfile.TemporaryDirectory() as d:
        m = _make_project(Path(d))
        assert m.tomli is not None


def test_load_cluster_paths_reads_user_folder():
    with tempfile.TemporaryDirectory() as d:
        m = _make_project(Path(d))
        cfg = m.load_cluster_paths()
        assert "cluster.local" in cfg
        assert cfg["cluster.local"]["output_dir"] == "/tmp/titan_test/out"


def test_load_cluster_paths_no_user_param():
    """The user/TITAN_USER dimension has been removed from the signature."""
    with tempfile.TemporaryDirectory() as d:
        m = _make_project(Path(d))
        assert "user" not in m.load_cluster_paths.__code__.co_varnames


def test_get_cluster_config():
    with tempfile.TemporaryDirectory() as d:
        m = _make_project(Path(d))
        cfg = m.get_cluster_config("local")
        assert cfg["output_dir"] == "/tmp/titan_test/out"
        assert cfg["cache_base"] == "/tmp/titan_test/cache"
        # Auto-generated cache subpaths derive from cache_base
        assert cfg["triton_cache"] == "/tmp/titan_test/cache/triton"


def test_get_tokenizer_path():
    with tempfile.TemporaryDirectory() as d:
        m = _make_project(Path(d))
        assert m.get_tokenizer_path("neox", "local") == "/tmp/titan_test/tokenizer/neox"


def test_get_paths():
    with tempfile.TemporaryDirectory() as d:
        m = _make_project(Path(d))
        paths = m.get_paths("test_dataset", "neox", "local")
        assert paths["cluster"] == "local"
        assert paths["tokenizer_path"] == "/tmp/titan_test/tokenizer/neox"
        assert paths["data_prefix"] == "/tmp/titan_test/data/train"
        assert paths["dataloader"] == "MMapDataset"


def test_get_env_exports():
    with tempfile.TemporaryDirectory() as d:
        m = _make_project(Path(d))
        exports = m.get_env_exports("local")
        assert 'export TRITON_CACHE_DIR="/tmp/titan_test/cache/triton"' in exports
        assert "TORCHINDUCTOR_FX_GRAPH_CACHE" in exports


def test_get_cli_args_uses_real_train_config():
    """get_cli_args resolves a real model train config and injects tokenizer/data args."""
    with tempfile.TemporaryDirectory() as d:
        m = _make_project(Path(d))
        args = m.get_cli_args(
            dataset="test_dataset",
            tokenizer="neox",
            cluster="local",
            config_file="qwen3_custom.toml",
            validate=False,
            project_root=str(REPO_ROOT),
        )
        assert "--job.config_file=" in args
        assert "--model.tokenizer_path=/tmp/titan_test/tokenizer/neox" in args


def test_missing_config_raises_filenotfound():
    with tempfile.TemporaryDirectory() as d:
        m = _make_project(Path(d), with_config=False)
        try:
            m.load_cluster_paths()
        except FileNotFoundError as e:
            assert "user/example" in str(e)  # points user at the template
        else:
            raise AssertionError("expected FileNotFoundError when user/cluster_paths.toml is absent")


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    import sys

    sys.exit(1 if _run_all() else 0)
