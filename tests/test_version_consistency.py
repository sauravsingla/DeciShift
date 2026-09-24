from pathlib import Path
import tomllib

import decishift


def test_runtime_version_matches_project_metadata():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert decishift.__version__ == pyproject["project"]["version"]
