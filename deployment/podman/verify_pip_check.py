"""Run ``pip check`` while allowing the known decord 0.6.0 wheel-tag defect."""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution


KNOWN_DECORD_WARNING = "decord 0.6.0 is not supported on this platform"
EXPECTED_DISTRIBUTION_VERSION = "0.1.3"
REQUIRED_DISTRIBUTION_FILES = (
    "generate.py",
    "scail2_inference/__init__.py",
    "scail2_inference/engine.py",
    "scail2_inference/media.py",
    "scail2_inference/runtime.py",
    "scail2_segments.py",
    "wan/__init__.py",
    "wan/scail.py",
)


def main() -> int:
    try:
        installed_distribution = distribution("scail2-inference")
    except PackageNotFoundError:
        print("scail2-inference is not installed", file=sys.stderr)
        return 1
    installed_version = installed_distribution.version
    if installed_version != EXPECTED_DISTRIBUTION_VERSION:
        print(
            "Unexpected scail2-inference version: "
            f"{installed_version}; expected {EXPECTED_DISTRIBUTION_VERSION}",
            file=sys.stderr,
        )
        return 1
    installed_files = {
        str(file) for file in (installed_distribution.files or ())
    }
    missing_files = [
        file_name
        for file_name in REQUIRED_DISTRIBUTION_FILES
        if file_name not in installed_files
    ]
    if missing_files:
        print(
            f"Missing installed files: {', '.join(missing_files)}",
            file=sys.stderr,
        )
        return 1

    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    messages = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    dependency_messages = [
        line for line in messages if not line.startswith("WARNING: The directory ")
    ]
    unexpected = [
        line for line in dependency_messages if line != KNOWN_DECORD_WARNING
    ]
    if result.returncode != 0 and (
        not dependency_messages
        or unexpected
        or KNOWN_DECORD_WARNING not in dependency_messages
    ):
        print("pip check reported unexpected dependency errors:", file=sys.stderr)
        print("\n".join(unexpected), file=sys.stderr)
        return result.returncode or 1

    if result.returncode != 0:
        print(f"Allowed known decord wheel-tag warning: {KNOWN_DECORD_WARNING}")
    else:
        print("No broken requirements found.")
    print(
        f"scail2-inference {installed_version} package and dependency check passed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
