#!/usr/bin/env python3
"""Generate a deterministic CycloneDX declared-dependency SBOM from pyproject.toml."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path
from typing import Any

NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+")


def _component(requirement: str, group: str) -> dict[str, Any]:
    match = NAME_RE.match(requirement)
    if match is None:
        raise ValueError(f"Cannot parse dependency requirement: {requirement}")
    name = match.group(0)
    return {
        "type": "library",
        "bom-ref": f"declared:{group}:{name.lower()}",
        "name": name,
        "properties": [
            {"name": "decishift:dependency-group", "value": group},
            {"name": "decishift:declared-requirement", "value": requirement},
        ],
    }


def build_sbom(pyproject: Path) -> dict[str, Any]:
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    project = data["project"]
    components: list[dict[str, Any]] = []
    for requirement in project.get("dependencies", []):
        components.append(_component(requirement, "runtime"))
    for group, requirements in sorted(project.get("optional-dependencies", {}).items()):
        for requirement in requirements:
            components.append(_component(requirement, f"optional:{group}"))
    components.sort(key=lambda item: item["bom-ref"])

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:00000000-0000-0000-0000-{project['version'].replace('.', '').zfill(12)}",
        "version": 1,
        "metadata": {
            "component": {
                "type": "library",
                "name": project["name"],
                "version": project["version"],
            },
            "properties": [
                {
                    "name": "decishift:sbom-scope",
                    "value": "declared direct dependencies and optional dependency groups; not a resolved transitive environment",
                }
            ],
        },
        "components": components,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyproject", default="pyproject.toml")
    parser.add_argument("--output", default="release-metadata/decishift-sbom.cdx.json")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_sbom(Path(args.pyproject)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
