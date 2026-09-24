#!/usr/bin/env python3
"""Verify that plugin.yaml version matches backend/constants.py __version__.

Run manually or as part of CI to catch version drift between the
Hermes plugin manifest and the Python package.
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Read version from constants.py
constants_py = os.path.join(ROOT, "backend", "constants.py")
py_version = None
with open(constants_py, encoding="utf-8") as fh:
    for line in fh:
        if line.startswith("__version__ = "):
            py_version = line.split("=", 1)[1].strip().strip('"\'')
            break
if py_version is None:
    print("FAIL: __version__ not found in backend/constants.py")
    sys.exit(1)

# Read version from plugin.yaml
plugin_yaml = os.path.join(ROOT, "plugin.yaml")
yaml_version = None
with open(plugin_yaml, encoding="utf-8") as fh:
    for line in fh:
        if line.startswith("version: "):
            yaml_version = line.split(":", 1)[1].strip()
            break
if yaml_version is None:
    print("FAIL: version not found in plugin.yaml")
    sys.exit(1)

if py_version != yaml_version:
    print(f"FAIL: version mismatch: constants.py={py_version}, plugin.yaml={yaml_version}")
    sys.exit(1)

print(f"OK: version {py_version} matches in both files")