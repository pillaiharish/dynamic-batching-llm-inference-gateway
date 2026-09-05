#!/usr/bin/env python3
"""Offline proof that laboratory identity changes plan destroy-before-create replacement."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

FAKE_ADAPTER = """\
import pathlib, sys
action, state = sys.argv[1], pathlib.Path(sys.argv[2])
with pathlib.Path("events").open("a") as events:
    events.write(action + "\\n")
if action == "create":
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("123\\n")
else:
    state.unlink()
"""


def run(terraform: str, directory: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [terraform, f"-chdir={directory}", *arguments],
        capture_output=True,
        text=True,
        check=True,
    )


def validate_replacement(terraform: str, source: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="vast-replacement-") as temporary:
        directory = Path(temporary)
        for path in source.glob("*.tf"):
            shutil.copy2(path, directory / path.name)
        (directory / "vast_lab.py").write_text(FAKE_ADAPTER, encoding="utf-8")
        common = ("-input=false", "-var=confirm_billable_resource=true", "-var=offer_id=1")
        run(terraform, directory, "init", "-backend=false", "-input=false")
        run(terraform, directory, "apply", "-auto-approve", *common, "-var=image=image:a")
        run(
            terraform,
            directory,
            "plan",
            "-out=replacement.tfplan",
            *common,
            "-var=image=image:b",
        )
        plan = json.loads(run(terraform, directory, "show", "-json", "replacement.tfplan").stdout)
        lab = next(
            change
            for change in plan["resource_changes"]
            if change["address"] == "terraform_data.h100_lab"
        )
        if lab["change"]["actions"] != ["delete", "create"]:
            raise AssertionError(f"unsafe replacement actions: {lab['change']['actions']}")
        run(terraform, directory, "apply", "-auto-approve", "replacement.tfplan")
        if (directory / "events").read_text().splitlines() != ["create", "destroy", "create"]:
            raise AssertionError(
                "replacement did not destroy the old lab before creating the new lab"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terraform", default=shutil.which("terraform"))
    args = parser.parse_args()
    if not args.terraform:
        parser.error("terraform executable not found; pass --terraform")
    validate_replacement(args.terraform, Path(__file__).parent)
    print("Vast laboratory replacement is destroy-before-create")


if __name__ == "__main__":
    main()
