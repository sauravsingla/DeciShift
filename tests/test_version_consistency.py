from pathlib import Path
import tomllib

import decishift


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_version_matches_project_metadata():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert decishift.__version__ == pyproject["project"]["version"]
